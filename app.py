import flask
import pickle
from config import config
from utils.utils import get_device, get_embedding, load_annoy_index, get_entail_scores
from flask import Flask, redirect, session, request
from model.model import SentenceTransformer, SequenceClassifier


app = Flask(__name__)

device = get_device()
# device = 'cpu' #Force cpu

model1 = SentenceTransformer().to(device)
model2 = SequenceClassifier().to(device)

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
with open('texts.pkl', 'rb') as f:
    texts = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

annoy_index = load_annoy_index('annoy_index', config.FEATURE_SIZE)

@app.route('/')
def status():
	return {
		"status": 200,
		"message": "Server Up" 
	}

@app.route('/predict', methods=['POST'])
def make_prediction():
	req_data = request.get_json()
	tweet_id = req_data['id']
	sentence = req_data['sentence']
	# sentences = clean_tweets(sentences)
	embedding = get_embedding(sentence)
	nns_ids = annoy_index.get_nns_by_vector(
		embedding,
		config.NEIGHBOURS, 
		search_k=config.SEARCH_K)

	simTexts = [texts[i] for i in nns_ids]
	simLabels = [labels[i] for i in nns_ids]

	scores = get_entail_scores(sentence, simTexts, model2, device)


	return {
		"status": 200,
		"data": scores
	}

if __name__ == "__main__":
	app.secret_key = 'DontSpreadMisinformation'
	app.run()

 


