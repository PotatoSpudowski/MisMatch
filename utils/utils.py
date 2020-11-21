import torch
import numpy as np
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
        output_list = []
        prob_list = [] 
        with torch.no_grad():
            for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)

                outputs = model(
                        ids=ids,
                        mask=mask,
                        )


                fin_targets.extend(targets)
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

        return fin_outputs, fin_targets

def build_annoy_index(features, feature_size, no_of_trees):
    annoy_index = AnnoyIndex(feature_size, metric='angular')
    for index, vector in tqdm(enumerate(features), total=len(features)):
        annoy_index.add_item(index, vector)
    annoy_index.build(no_of_trees)

    return annoy_index
