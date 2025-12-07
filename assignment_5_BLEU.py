from sacrebleu import corpus_bleu

ref_path = "toy_example/data/raw/test.en"
refs = [open(ref_path, encoding="utf-8").read().splitlines()]

for alpha in [0, 0.3, 0.5, 0.7]:
    hyp_path = f"toy_example/output_alpha_{alpha}.en"
    with open(hyp_path, encoding="utf-8") as f:
        hyps = f.read().splitlines()
    score = corpus_bleu(hyps, refs)
    print(f"alpha={alpha}: BLEU = {score.score:.2f}")
