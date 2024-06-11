import os
import glob
import wandb
import argparse
import random
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import make_dataset, load_data
from optimizer import StochasticGradientLangevinDynamics


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate_start", type=float, default=5e-4)
parser.add_argument("--learning_rate_end", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--data_path", type=str, default="./data/Training_Dataset_Cropped_Split/")
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
parser.add_argument("--message", type=str, default="")
args = parser.parse_args()


if __name__ == "__main__":

    # Set random seeds for reproducibility of weights initialization
    # All three of these must be set in order to make the weights initialization reproducible
    #random.seed(args.random_seed)
    #tf.random.set_seed(args.random_seed)
    #np.random.seed(args.random_seed)
    tf.keras.utils.set_random_seed(args.random_seed)
    
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(project=args.project, name=timestr, config=vars(args))

    X_train, y_train = load_data(args.data_path + '/train/')
    X_val, y_val = load_data(args.data_path + '/val/')
    
    ds_train = make_dataset(X_train, y_train, args.image_size, args.batch_size, shuffle=True, seed=args.random_seed)
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=args.random_seed)
    
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
    if args.apply_brightness:
        layers.append(tf.keras.layers.RandomBrightness(0.2, seed=args.random_seed))
    if args.apply_contrast:
        layers.append(tf.keras.layers.RandomContrast(0.4, seed=args.random_seed))
    
    layers.append(base_model)
    layers.append(tf.keras.layers.Dense(4, activation='softmax'))
    
    model = tf.keras.Sequential(layers)
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate_start,
        end_learning_rate=args.learning_rate_end,
        decay_steps=args.epochs*len(ds_train),
        power=2.0,
        )        
    
    sample_size = len(X_train)
    
    optimizer = StochasticGradientLangevinDynamics(
        learning_rate=lr_schedule,
        rho=0.9,
        epsilon=1e-7,
        burnin=np.ceil(sample_size / args.batch_size).astype(int) * args.epochs,
        data_size=sample_size,
        weight_decay=args.weight_decay,
        )

    # also monitor the learning rate
    lr_variable = wandb.define_metric('learning_rate')
    def log_lr(epoch, logs):
        wandb.log({'learning_rate': optimizer._decayed_lr(tf.float32).numpy()}, commit=False)
    lr_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_lr)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        ds_train, 
        epochs=args.epochs, 
        validation_data=ds_val, 
        callbacks=[wandb.keras.WandbCallback(save_model=False), lr_callback]
        )
    
    os.makedirs(args.save_path, exist_ok=True)
    model.save(args.save_path + f'/{timestr}.keras')
    