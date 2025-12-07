import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.decode import decode
from seq2seq import models, utils


def get_args():
    """
    Defines generation-specific command-line arguments.
    """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')

    # Runtime options
    parser.add_argument('--cuda', action='store_true', help='Use GPU for decoding')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    # Input / output
    parser.add_argument('--input', required=True, help='Input file (one source sentence per line)')
    parser.add_argument('--src-tokenizer', required=True, help='Path to source sentencepiece tokenizer')
    parser.add_argument('--tgt-tokenizer', required=True, help='Path to target sentencepiece tokenizer')
    parser.add_argument('--checkpoint-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size for decoding')
    parser.add_argument('--output', required=True, help='Output file for translations')
    parser.add_argument('--max-len', default=128, type=int, help='Maximum generation length')

    # BLEU evaluation
    parser.add_argument('--bleu', action='store_true', help='Compute BLEU after translation')
    parser.add_argument('--reference', help='Reference file (required if --bleu is set)')

    return parser.parse_args()


def main(args):
    """
    Main translation and evaluation routine.
    """
    torch.manual_seed(args.seed)

    # Load checkpoint and restore training arguments
    state_dict = torch.load(
        args.checkpoint_path,
        map_location=lambda s, l: default_restore_location(s, 'cpu'),
        weights_only=False
    )
    args = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    utils.init_logging(args)

    # Load tokenizers
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    # Build model
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])

    logging.info(f'Loaded model from {args.checkpoint_path}')

    # Read input sentences
    with open(args.input, encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]

    # Encode source sentences and append EOS
    src_encoded = [
        torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True))
        for line in src_lines
    ]

    # Truncate inputs to maximum allowed length
    max_src_len = min(model.encoder.pos_embed.size(1), args.max_len)
    src_encoded = [s[:max_src_len] for s in src_encoded]

    DEVICE = 'cuda' if args.cuda else 'cpu'
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()

    # Clear output file
    with open(args.output, 'w', encoding='utf-8'):
        pass

    def postprocess_ids(ids):
        """
        Remove BOS, truncate at first EOS, and remove PAD tokens.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and ids[0] == BOS:
            ids = ids[1:]
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        return [i for i in ids if i != PAD]

    translations = []
    output_lengths = []  # Stores decoded sentence lengths (in tokens)

    make_batch = utils.make_batch_input(
        device=DEVICE,
        pad=PAD,
        max_seq_len=args.max_len
    )

    def batch_iter(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    start_time = time.perf_counter()

    # Translation loop
    for batch in tqdm(batch_iter(src_encoded, args.batch_size)):
        with torch.no_grad():
            # Pad batch to uniform length
            max_len = max(len(x) for x in batch)
            batch_padded = [
                torch.cat([x, torch.full((max_len - len(x),), PAD, dtype=torch.long)])
                if len(x) < max_len else x
                for x in batch
            ]
            src_tokens = torch.stack(batch_padded).to(DEVICE)
            dummy_y = torch.full_like(src_tokens, PAD)

            # Create masks
            src_tokens, _, _, src_pad_mask, _ = make_batch(src_tokens, dummy_y)

            # Decode without teacher forcing
            predictions = decode(
                model=model,
                src_tokens=src_tokens,
                src_pad_mask=src_pad_mask,
                max_out_len=args.max_len,
                tgt_tokenizer=tgt_tokenizer,
                args=args,
                device=DEVICE
            )

        # Convert token IDs to text
        for sent in predictions:
            clean_ids = postprocess_ids(sent)
            translation = tgt_tokenizer.Decode(clean_ids)

            translations.append(translation)
            output_lengths.append(len(clean_ids))

            with open(args.output, 'a', encoding='utf-8') as out_f:
                out_f.write(translation + '\n')

    end_time = time.perf_counter()

    logging.info(f'Wrote {len(translations)} lines to {args.output}')
    logging.info(f'Translation completed in {end_time - start_time:.2f} seconds')

    # Report average output sentence length
    avg_len = float(np.mean(output_lengths))
    print(f'Average output length: {avg_len:.2f} tokens')

    # Compute BLEU score if requested
    if args.bleu:
        with open(args.reference, encoding='utf-8') as ref_f:
            references = [line.strip() for line in ref_f if line.strip()]
        bleu = sacrebleu.corpus_bleu(translations, [references])
        print(f'BLEU score: {bleu.score:.2f}')


if __name__ == '__main__':
    args = get_args()
    if args.bleu and not args.reference:
        raise ValueError('You must provide --reference when using --bleu.')
    main(args)
