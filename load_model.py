"""
Training script.

Created on 2019-01-28-12-21
Author: Stephan Rasp, raspstephan@gmail.com
"""

from cbrain.imports import *
from cbrain.utils import *
from cbrain.losses import *
from cbrain.models_vae import VariationalAutoEncoder


os.environ["CUDA_VISIBLE_DEVICES"] = 'None'
limit_mem()
# Hard coded
vae = VariationalAutoEncoder(original_dim = 65, intermediate_dim = [64] ,latent_dim = 12, activation="LeakyReLU")

# Hard coded
model_fn = 'saved_models/001_VAE_/test/'
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.train_on_batch(np.array([[0.]*65]))
vae.load_weights(model_fn)
vae.summary()




