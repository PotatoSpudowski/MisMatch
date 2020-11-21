import pandas as pd
import torch
import pickle
from config import config
from utils.utils import get_device, generate_embeddings, build_annoy_index
from model.model import SentenceTransformer
from data.data import SentenceTransformerDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = get_device()
# device = 'cpu'

df = pd.read_csv('inputs/data.csv')
texts = df['text'].values
labels = df['label'].values

model = SentenceTransformer().to(device)

batch_dataset = SentenceTransformerDataset(
                    text=texts,
                    target=labels)

batch_data_loader = torch.utils.data.DataLoader(
        batch_dataset,
        sampler = SequentialSampler(batch_dataset),
        batch_size=config.BATCH_SIZE,
        num_workers=4
    )

embeddings, labels = generate_embeddings(batch_data_loader, model, device)

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

annoy_index = build_annoy_index(
    embeddings, 
    config.MAXLEN, 
    config.TREES)

annoy_index.save("annoy_index")


