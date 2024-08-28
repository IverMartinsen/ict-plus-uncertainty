import os
import glob
import json
import wandb
import argparse
import random
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import make_dataset, load_data
from swag_utils import SWAGDiagonalCallback
from optimizer import StochasticGradientLangevinDynamics
from schedule import PolynomialDecay

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate_start", type=float, default=5e-4)
parser.add_argument("--learning_rate_end", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--data_path_train", type=str, default="./data/Training_Dataset_Cropped_Split/")
parser.add_argument("--data_path_val", type=str, default=None)
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--project", type=str, default="ict-plus-uncertainty")
parser.add_argument("--save_path", type=str, default="./models/")
parser.add_argument("--apply_flip", type=bool, default=True)
parser.add_argument("--apply_translation", type=bool, default=False)
parser.add_argument("--apply_rotation", type=bool, default=True)
parser.add_argument("--apply_zoom", type=bool, default=True)
parser.add_argument("--apply_brightness", type=bool, default=False)
parser.add_argument("--apply_contrast", type=bool, default=False)
parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--path_to_weights", type=str, default=None)
parser.add_argument("--message", type=str, default="")
parser.add_argument("--swag", type=bool, default=False)
parser.add_argument("--sgld", type=bool, default=False)
parser.add_argument("--burnin", type=int, default=1e8)
args = parser.parse_args()


if __name__ == "__main__":
    
    os.makedirs(args.save_path, exist_ok=True)

    # Set random seeds for reproducibility of weights initialization
    # All three of these must be set in order to make the weights initialization reproducible
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(project=args.project, name=timestr, config=vars(args))

    X_train, y_train = load_data(args.data_path_train)
    ds_train = make_dataset(X_train, y_train, args.image_size, args.batch_size, shuffle=True, seed=args.random_seed)
    if args.data_path_val is None:
        ds_val = None
    else:
        X_val, y_val = load_data(args.data_path_val)
        ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=args.random_seed)
    
    if args.path_to_weights is not None:
        print(f"Loading model from {args.path_to_weights}")
        model = tf.keras.models.load_model(
            args.path_to_weights, custom_objects={
                'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
                'PolynomialDecay': PolynomialDecay,
                })
        model.optimizer.learning_rate = args.learning_rate_end
    else:
        print("Building new model")
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, pooling='avg')
        wandb.config.update({'base_model': base_model.name})
        
        layers = []
        
        if args.apply_flip:
            layers.append(tf.keras.layers.RandomFlip(seed=args.random_seed))
        if args.apply_rotation:
            layers.append(tf.keras.layers.RandomRotation(0.2, seed=args.random_seed))
        if args.apply_translation:
            layers.append(tf.keras.layers.RandomTranslation(0.2, 0.2, seed=args.random_seed))
        if args.apply_zoom:
            layers.append(tf.keras.layers.RandomZoom((-0.2, 0.0), seed=args.random_seed))
        if args.apply_contrast:
            layers.append(tf.keras.layers.RandomContrast(0.4, seed=args.random_seed))
        
        layers.append(base_model)
        layers.append(tf.keras.layers.Dense(4, activation='softmax'))
        
        model = tf.keras.Sequential(layers)
        
        lr_schedule = PolynomialDecay(
            initial_learning_rate=args.learning_rate_start,
            end_learning_rate=args.learning_rate_end,
            decay_steps=args.epochs*len(ds_train),
            power=2.0,
            warmup_steps=10*len(ds_train),
            )        
        
        sample_size = len(X_train)
        
        optimizer = StochasticGradientLangevinDynamics(
            learning_rate=lr_schedule,
            rho=0.9,
            epsilon=1e-7,
            burnin=args.burnin,
            data_size=sample_size,
            weight_decay=args.weight_decay,
        )
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = []
    # Log the training
    callbacks.append(wandb.keras.WandbCallback(save_model=False))
    # Log the learning rate
    lr_variable = wandb.define_metric('learning_rate')
    def log_lr(epoch, logs):
        wandb.log({'learning_rate': model.optimizer._decayed_lr(tf.float32).numpy()}, commit=False)
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=log_lr))
    # If using SWAG, save the diagonal mean and variance
    if args.swag:
        os.makedirs(os.path.join(args.save_path, timestr), exist_ok=True)
        callbacks.append(SWAGDiagonalCallback(model, os.path.join(args.save_path, timestr)))
    # If using SGLD, save the model at the end of each epoch
    if args.sgld:
        os.makedirs(os.path.join(args.save_path, timestr), exist_ok=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.path.join(args.save_path, timestr), "{epoch:02d}.keras"),
            save_best_only=False,
            save_weights_only=False,
        )
        sample_size = len(X_train)
        steps_per_epoch = np.ceil(sample_size / args.batch_size).astype(int)
        burnin = steps_per_epoch*100 # number of training steps before starting to add gradient noise
        model.optimizer._burnin = tf.convert_to_tensor(0, name='burnin')

    json.dump(vars(args), open(os.path.join(args.save_path, f'{timestr}.json'), 'w'))

    model.fit(
        ds_train, 
        epochs=args.epochs, 
        validation_data=ds_val, 
        callbacks=callbacks,
        )

    model.save(args.save_path + f'/{timestr}.keras')
