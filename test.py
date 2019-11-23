from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

vae_filepath = 'saved_models/001_VAE_/model_vae.h5'

vae = load_model(vae_filepath, custom_objects={'LeakyReLU': LeakyReLU})