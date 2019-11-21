"""
Training script.

Created on 2019-01-28-12-21
Author: Stephan Rasp, raspstephan@gmail.com
"""

from cbrain.imports import *
from cbrain.utils import *
from cbrain.losses import *
from cbrain.data_generator_vae import DataGenerator
from cbrain.models_new import *
from cbrain.learning_rate_schedule import LRUpdate
from cbrain.save_weights import save2txt, save_norm
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import mse
import json
from datetime import datetime,date
from cbrain.models_vae import VariationalAutoEncoder

def main(args):
    """Main training script."""

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    limit_mem()

    logging.basicConfig(
    filename='saved_models\logs\model_{}_{}.txt'.format(args.exp_name,date.today()),
    filemode ='a',
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
    )
            


    # Load output scaling dictionary
    out_scale_dict = load_pickle(args.output_dict)

    logging.info('Create training and validation data generators')
    train_gen = DataGenerator(
        data_fn=args.data_dir + args.train_fn,
        input_vars=args.inputs,
        output_vars=args.outputs,
        norm_fn=args.data_dir + args.norm_fn,
        input_transform=(args.input_sub, args.input_div),
        output_transform=out_scale_dict,
        batch_size=args.batch_size,
        shuffle=True,
        var_cut_off=args.var_cut_off
    )

    if args.valid_fn is not None:
        valid_gen = DataGenerator(
            data_fn=args.data_dir + args.valid_fn,
            input_vars=args.inputs,
            output_vars=args.outputs,
            norm_fn=args.data_dir + args.norm_fn,
            input_transform=(args.input_sub, args.input_div),
            output_transform=out_scale_dict,
            batch_size=args.batch_size * 10,
            shuffle=False,
            var_cut_off=args.var_cut_off
        )
    else:
        valid_gen = None

    logging.info('Build model')


## change from here
    '''
    vae = vae_model(
        input_shape= train_gen.n_inputs,
        latent_dim = args.latent_dim,
        encoder_layers = args.encoder_layers,
        decoder_layers = args.decoder_layers,
        activation = args.activation)
    '''
    lrs = LearningRateScheduler(LRUpdate(args.lr, args.lr_step, args.lr_divide))
    
    vae = VariationalAutoEncoder(original_dim = train_gen.n_inputs, intermediate_dim = args.intermediate_dim ,latent_dim = args.latent_dim, activation=args.activation)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()
    #vae.compile(optimizer, loss=mse_loss_fn, metr)

    
    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #epochs = 3

    #vae = VariationalAutoEncoder(784, 64, 32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    #vae.fit(x_train, x_train, epochs=3, batch_size=64) 

    vae.fit_generator(
        train_gen, epochs=args.epochs, validation_data=valid_gen, callbacks=[lrs]
        )

    '''
    def mse_loss(y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)
        reconstruction_loss *= 64 # vae.get_layer('encoder_input').input_shape
        z_mean = vae.get_layer('encoder').get_layer('z_mean').output
        z_log_var = vae.get_layer('encoder').get_layer('z_log_var').output
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)


    print(vae.summary())

    logging.info('Compile model')
    #loss = args.loss
    metrics = [mse]

    vae.compile(args.optimizer, loss=mse_loss, metrics=metrics)

    lrs = LearningRateScheduler(LRUpdate(args.lr, args.lr_step, args.lr_divide))

    logging.info('Train model')
    vae.fit_generator(
        train_gen, epochs=args.epochs, validation_data=valid_gen, callbacks=[lrs]
        )
    '''
    #print(np.array([x[1] for x in valid_gen]).flatten())
    #print(model.predict_generator(valid_gen).flatten())
    # save valid(dev)/ prediction    
    #print(len([x[0][0] for x in valid_gen][0]))
    #pd.DataFrame({'Actual':np.array([x[1] for x in valid_gen]).flatten()/out_scale_dict['PRECT'],
    #            'Predicted':model.predict_generator(valid_gen).flatten()/out_scale_dict['PRECT']}).to_csv('saved_models\logs\prediction.csv')

    if args.exp_name is not None:
        exp_dir = args.model_dir + args.exp_name + '/'
        os.makedirs(exp_dir, exist_ok=True)
        model_fn = exp_dir + 'model_vae'
        logging.info(f'Saving model as {model_fn}')
        #vae.save(model_fn)
        tf.saved_model.save(vae, exp_dir + "/" + "test/")
        '''
        if args.save_txt:
            weights_fn = exp_dir + 'vae_weights'
            logging.info(f'Saving weights as {weights_fn}')
            vae.save_weights(weights_fn)
            save2txt(weights_fn, exp_dir)
        '''
        save_norm(train_gen.input_transform, train_gen.output_transform, exp_dir)
        
    logging.info('Done!')


# Create command line interface
if __name__ == '__main__':
    p = ArgParser()
    p.add('-c', '--config_file', default='config.yml', is_config_file=True, help='Path to config file.')

    # Data arguments
    p.add('--data_dir', type=str, help='Path to preprocessed data files.')
    p.add('--inputs', type=str, nargs='+', help='List of input variables.')
    p.add('--outputs', type=str, nargs='+', help='List of output variables.')
    p.add('--train_fn', type=str, help='File name of training file.')
    p.add('--norm_fn', type=str, help='File name of normalization file.')
    p.add('--input_sub', type=str, help='What to subtract from input array. E.g. "mean"')
    p.add('--input_div', type=str, help='What to divide input array by. E.g. "maxrs"')
    p.add('--output_dict', type=str, help='Output scaling dictionary.')
    p.add('--var_cut_off', type=json.loads, help='Input variable cut off for upper levels.')
    p.add('--latent_dim', type = int, default = 32, help='Size of the latent vector')
    p.add('--intermediate_dim', type = int, nargs='+', help='Intermediate layer sizes.')
    #p.add('--decoder_layers', type = int, nargs='+', help='Decoder layer sizes.')

    p.add('--valid_fn', type=str, default=None, help='File name of training file.')

    # Neural network hyperparameteris
    p.add('--batch_size', type=int, default=1024, help='Batch size of training generator.')
    #p.add('--hidden_layers', type=int, nargs='+', help='Hidden layer sizes.')
    p.add('--activation', type=str, default='LeakyReLU', help='Activation function.')
    p.add('--optimizer', type=str, default='adam', help='Optimizer.')
    #p.add('--conservation_layer', dest='conservation_layer', action='store_true', help='Add conservation layer.')
    #p.set_defaults(conservation_layer=False)

    # Loss parameters
    p.add('--loss', type=str, default='mse', help='Loss function.')
    #p.add('--conservation_metrics', dest='conservation_metrics', action='store_true', help='Add conservation metrics.')
    #p.set_defaults(conservation_metrics=False)
    #p.add('--alpha_mass', type=float, default=0.25, help='If weak_loss, weight of mass loss.')
    #p.add('--alpha_ent', type=float, default=0.25, help='If weak_loss, weight of ent loss.')
    #p.add('--noadiab', dest='noadiab', action='store_true',help='noadiab')
    #p.set_defaults(noadiab=False)

    # Learning rate schedule
    p.add('--lr', type=float, default=0.001, help='Initial learning rate.')
    p.add('--lr_step', type=int, default=2, help='Divide every step epochs.')
    p.add('--lr_divide', type=float, default=5, help='Divide by this number.')
    p.add('--epochs', type=int, default=10, help='Number of epochs.')

    # Save parameters
    p.add('--exp_name', type=str, default=None, help='Experiment identifier.')
    p.add('--model_dir', type=str, default='./saved_models/', help='Model save path.')
    p.add('--save_txt', dest='save_txt', action='store_true', help='Save F90 txt files.')
    p.set_defaults(save_txt=True)

    p.add('--gpu', type=str, default=None, help='Which GPU to use.')

    args = p.parse_args()
    main(args)
