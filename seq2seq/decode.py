import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


# Beam search decoding function for sequence-to-sequence models
def beam_search_decode(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 4
) -> list:
    """
    Performs beam search decoding for a batch of input sequences.
    Returns a list of predicted token sequences (one per example in the batch).
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # To keep things simple, we process each example in the batch independently.
    predicted_tokens = []
    for b in range(batch_size):
        # Each beam is a tuple: (sequence_so_far, cumulative_logprob, finished)
        beams = [
            (torch.tensor([BOS], dtype=torch.long, device=device), 0.0, False)
        ]
        for t in range(max_out_len):
            new_beams = []
            for seq, score, finished in beams:
                if finished:
                    # If already finished (EOS produced), just continue
                    new_beams.append((seq, score, True))
                    continue
                # Prepare model input: (1, seq_len)
                generated = seq.unsqueeze(0)
                # Pad mask for decoder
                max_len = model.decoder.pos_embed.size(1)
                if generated.size(1) > max_len:
                    generated = generated[:, :max_len]
                trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)
                # Model forward
                out = model(
                    src_tokens[b:b+1],  # (1, src_len)
                    src_pad_mask[b:b+1] if src_pad_mask is not None else None,
                    generated,
                    trg_pad_mask
                ).to(device)
                # Logits for last step
                next_token_logits = out[:, -1, :]  # (1, vocab)
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                # Get top beam_size candidates
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    next_token = topk_tokens[0, k].item()
                    next_logprob = topk_log_probs[0, k].item()
                    # Append next token
                    new_seq = torch.cat([seq, torch.tensor([next_token], device=device, dtype=torch.long)], dim=0)
                    end = (next_token == EOS)
                    new_beams.append((new_seq, score + next_logprob, end))
            # Keep only top beam_size beams (by score)
            # Sort by cumulative logprob (descending)
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams
            # If all beams are finished, stop early
            if all(f for _, _, f in beams):
                break
        # Select the best beam (highest score, finished if possible)
        finished_beams = [b for b in beams if b[2]]
        if finished_beams:
            best_beam = max(finished_beams, key=lambda x: x[1])
        else:
            best_beam = beams[0]
        # Remove BOS and anything after EOS
        seq = best_beam[0].tolist()
        if seq[0] == BOS:
            seq = seq[1:]
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens
