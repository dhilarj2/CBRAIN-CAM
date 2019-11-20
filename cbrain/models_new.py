"""
Define all different types of models.

Created on 2019-01-28-13-17
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *
from .cam_constants import *
from tensorflow.keras.layers import *
from .layers import *

##from keras.layers import Lambda, Input, Dense
##from keras.models import Model
##from keras.losses import mse, binary_crossentropy
from keras import backend as K
import numpy as np
import os


def act_layer(act):
    """Helper function to return regular and advanced activation layers"""
    act = Activation(act) if act in tf.keras.activations.__dict__.keys() \
        else tf.keras.layers.__dict__[act]()
    return act


def fc_model(input_shape, output_shape, hidden_layers, activation, conservation_layer=False,
             inp_sub=None, inp_div=None, norm_q=None):
    inp = Input(shape=(input_shape,))

    # First hidden layer
    x = Dense(hidden_layers[0])(inp)
    x = act_layer(activation)(x)

    # Remaining hidden layers
    for h in hidden_layers[1:]:
        x = Dense(h)(x)
        x = act_layer(activation)(x)

    if conservation_layer:
        x = SurRadLayer(inp_sub, inp_div, norm_q)([inp, x])
        x = MassConsLayer(inp_sub, inp_div, norm_q)([inp, x])
        out = EntConsLayer(inp_sub, inp_div, norm_q)([inp, x])

    else:
        out = Dense(output_shape)(x)
        out = act_layer('relu')(out)
        #out = 10**out
    return tf.keras.models.Model(inp, out)




#### VAE MODEL

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_model(input_shape,latent_dim, encoder_layers, decoder_layers,activation='relu'):
    
    output_shape = input_shape
    #decoder_layers = encoder_layers

    # a layer instance is callable on a tensor, and returns a tensor
    # set up the model layers 
    inputs = Input(shape=input_shape, name='encoder_input')
    # First hidden layer
    x = Dense(encoder_layers[0], activation = act_layer(activation))(inputs)
    #x = Dense(20, activation = 'relu')(inputs)
    #x = act_layer(activation)(x)

    for h in encoder_layers[1:]:
        x = Dense(h, activation = act_layer(activation))(x)
        x = act_layer(activation)(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = tf.keras.models.Model(inputs = inputs,outputs = [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())
    #plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)


    # Hidden layer
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    # Decoder
    x = Dense(decoder_layers[0])(latent_inputs)
    x = act_layer(activation)(x)

    for h in decoder_layers[1:]:
        x = Dense(h)(x)
        x = act_layer(activation)(x)

    outputs = Dense(output_shape, activation='linear')(x) ##check this!!


    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    print(decoder.summary())

    outputs = decoder(encoder(inputs)[2])

    return tf.keras.models.Model(inputs, outputs, name='vae')
    




    




