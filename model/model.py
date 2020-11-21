import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from config import config


class SentenceTransformer(torch.nn.Module):
    def __init__(self):
        super(SentenceTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(config.MODEL1)

  
    def forward(self, ids, mask):
        bo, po = self.bert(
            ids, 
            attention_mask=mask,
        )
        
        # output = bo[0][:,0,:] #CLS token embedding
        output = po #Pooled embedding

        return output
