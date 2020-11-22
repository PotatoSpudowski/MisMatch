# MisMatch

Checking missinformation service

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