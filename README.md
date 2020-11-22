# MisMatch

Checking missinformation service

A Webapp that receives tweets as input and identifies if a similar tweet has been flagged as False or True.

We generate embeddings of all the previously flagged Tweets/Text data and create an Approximate nearest neighbor Index that we can use to retrieve similar embeddings, given an input embedding.

From the nearest neighbors, we can get similar tweets and we compare our input tweet with similar tweets using a model similar to the zero-shot classification model. The model tells us if the inputs contradict or not. If they do not and we have a high score, we can assume that the inputs are similar and can assign the same label to the query tweet. The label corresponding to the tweet combination with the highest score is returned.

## Return Body
![Api Call](https://github.com/PotatoSpudowski/MisMatch/blob/main/images/2.png)

## Requirements
Python 3.7.6

### Enviroment

1. (Recommended) Create a virtual environment:
```
python -m virtualenv env
```
Enter the virtual environment.
```
source env/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

## Run service
To run the service just run the command:
```
python app.py
```
To re index new Tweets/Texts add them in the Inputs folder and run:
```
python train.py
```
