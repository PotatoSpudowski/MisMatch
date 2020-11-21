import torch
from config import config

class SentenceTransformerDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER1
        self.max_len = config.MAXLEN
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())


        inputs = self.tokenizer.encode_plus(
                                    text,
                                    add_special_tokens=True,
                                    max_length=self.max_len,
                                    padding='max_length',
                                    return_token_type_ids=True,
                                    truncation=True
                                    )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': self.target[item]
        }