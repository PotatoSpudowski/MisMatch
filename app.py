import flask

from config import config
from flask import Flask, redirect, session, request
from model import model


app = Flask(__name__)

# device = get_device()

print(model.test())
# device = 'cpu' #Force cpu


@app.route('/')
def status():
	return {
		"status": 200,
		"message": "Server Up" 
	}

if __name__ == "__main__":
	app.secret_key = 'DontSpreadMisinformation'
	app.run()

 


