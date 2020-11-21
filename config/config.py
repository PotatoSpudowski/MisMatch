from transformers import AutoTokenizer

BATCH_SIZE = 64
MAXLEN = 280

TREES = 100
FEATURE_SIZE = 768

MODEL1 = "sentence-transformers/roberta-base-nli-stsb-mean-tokens"
MODEL2 = "valhalla/distilbart-mnli-12-6"

TOKENIZER1 = AutoTokenizer.from_pretrained("sentence-transformers/roberta-base-nli-stsb-mean-tokens")
TOKENIZER2 = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-6")