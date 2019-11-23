import tensorflow as tf
from tensorflow.keras import layers

from .imports import *
from .cam_constants import *
# from tensorflow.keras.layers import *
# from .layers import *
import tensorflow as tf
##from keras.layers import Lambda, Input, Dense
##from keras.models import Model
##from keras.losses import mse, binary_crossentropy
from keras import backend as K
import numpy as np
import os

from tensorflow.keras.layers import *
from .layers import *


def act_layer(act):
    """Helper function to return regular and advanced activation layers"""
    act = Activation(act) if act in tf.keras.activations.__dict__.keys() \
        else tf.keras.layers.__dict__[act]()
    return act


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, intermediate_dim=[64], activation="relu", name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = {}
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.activation = activation

        self.number_of_layers = len(self.intermediate_dim)
        i = 0
        self.dense_proj[str(i)] = layers.Dense(self.intermediate_dim[0], activation=act_layer(self.activation))

        for h in self.intermediate_dim[1:]:
            i += 1
            self.dense_proj[str(i)] = layers.Dense(h, activation=act_layer(self.activation))  # (self.dense_proj[i-1])

        self.dense_mean = layers.Dense(self.latent_dim)
        self.dense_log_var = layers.Dense(self.latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj[str(0)](inputs)
        for i in range(1, self.number_of_layers):
            x = self.dense_proj[str(i)](x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim=64, intermediate_dim=[64], activation="relu", name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = {}
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.number_of_layers = len(self.intermediate_dim)
        self.original_dim = original_dim
        # self.dense_proj[str(i)] = layers.Dense(self.intermediate_dim[0], activation = act_layer(self.activation))

        for i, h in enumerate(self.intermediate_dim):
            self.dense_proj[str(i)] = layers.Dense(h, activation=act_layer(self.activation))  # (self.dense_proj[i-1])

        self.dense_output = layers.Dense(original_dim, activation='relu')

    def call(self, inputs):
        x = self.dense_proj[str(0)](inputs)
        for i in range(1, self.number_of_layers):
            x = self.dense_proj[str(i)](x)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=[64], latent_dim=32, name='autoencoder', activation="relu",
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        # self,latent_dim=32,intermediate_dim=[64],name='encoder',activation = "relu"
        self.encoder = Encoder(latent_dim, intermediate_dim, activation)
        # self,original_dim=64,intermediate_dim=[64],name='decoder',activation = "relu"
        self.decoder = Decoder(original_dim, intermediate_dim, activation)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed