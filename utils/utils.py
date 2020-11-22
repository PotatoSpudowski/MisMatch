import torch
import numpy as np
from config import config
from annoy import AnnoyIndex
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device

def generate_embeddings(data_loader, model, device): 
        model.eval()
        fin_targets = []
        fin_outputs = []
        fin_texts = []
        with torch.no_grad():
            for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]
                text = d["text"]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)

                outputs = model(
                        ids=ids,
                        mask=mask,
                        )


                fin_targets.extend(targets)
                fin_texts.extend(text)
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

        return fin_outputs, fin_texts, fin_targets

def get_embedding(sentence, model, device):
    tokenizer = config.TOKENIZER1
    inputs = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=config.MAXLEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
        )
    
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    output = model(
                ids=ids,
                mask=mask,
                )
    
    return output[0]

def build_annoy_index(features, feature_size, no_of_trees):
    annoy_index = AnnoyIndex(feature_size, metric='angular')
    for index, vector in tqdm(enumerate(features), total=len(features)):
        annoy_index.add_item(index, vector)
    annoy_index.build(no_of_trees)

    return annoy_index

def load_annoy_index(path, feature_size):
    annoy_index = AnnoyIndex(feature_size, metric='angular')
    annoy_index.load(path)

    return annoy_index

def get_entail_scores(inputText, simTexts, model, device):
    tokenizer = config.TOKENIZER2
    SeqPairs = [(inputText, simTexts[i]) for i in range(len(simTexts))]
    inputs = tokenizer(SeqPairs, 
                       padding=True, 
                       truncation=True, 
                       return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    logits = model(ids=input_ids, mask=attention_mask)[0]
    entail_contr_logits = np.array([logits[:,0], logits[:,2]])
    outputs = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
    outputs = [o.cpu().detach().numpy() for o in outputs][1]

    return outputs

