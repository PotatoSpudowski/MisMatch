from transformers import AutoTokenizer

BATCH_SIZE = 64
MAXLEN = 280

TREES = 100
FEATURE_SIZE = 768

NEIGHBOURS = 5
SEARCH_K = 100

MODEL1 = "sentence-transformers/roberta-base-nli-stsb-mean-tokens"
# MODEL2 = "valhalla/distilbart-mnli-12-6"
MODEL2 = "facebook/bart-large-mnli"


TOKENIZER1 = AutoTokenizer.from_pretrained("sentence-transformers/roberta-base-nli-stsb-mean-tokens")
# TOKENIZER2 = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-6")
TOKENIZER2 = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")