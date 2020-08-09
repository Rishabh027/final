import base64
import numpy as np 
import io
from PIL import Image
import keras
from keras import backend as k 
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
	global model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk") 

def preprocess_image(image, target_size):
	if image.mode != "L":
		image = image.covert("L")
		image = image.resize(target_size)
		image = np.expand_dims(image,axis=0)
		image /=255
	return image	




@app.route("/predict", methods=["GET","POST"])
def prediction():
	message = request.get_json(force=True)
	encoded	= message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	process_image = preprocess_image(image,target_size=(48,48))
	predic = model.prediction(process_image).tolist()

	response = {
		'prediction':
		{
			'anger': predic[0][0],
			'disgust': predic[0][1],
			'fear': predic[0][2],
			'happy': predic[0][3],
			'sad': predic[0][4],
			'surprise': predic[0][5],
			'neutral' : predic[0][1]
		}
	}
	return jsonify(response)	

if __name__ = "__main__":
	print("loading keras model")
	get_model()
	app.run()
