from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import model_from_json
import json


#with open('saved_models/001_VAE_/model_vae.json') as json_file:
#    json_config = json_file.read()

#model_from_json(json_config,custom_objects={'LeakyReLU':LeakyReLU},compile = False)

#vae_loaded = load_model("saved_models/001_VAE_/model_vae.h5") #,compile = False,custom_objects={'LeakyReLU':LeakyReLU})


import tensorflow as tf
x = tf.compat.v2.saved_model.load(export_dir = "saved_models/001_VAE_/test/")