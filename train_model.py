import os
import glob
import wandb
import argparse
import random
import tensorflow as tf
import numpy as np
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate_start", type=float, default=5e-4)
parser.add_argument("--learning_rate_end", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--data_path", type=str, default="./data/Training_Dataset_Cropped_Split/")
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--project", type=str, default="ict-plus-uncertainty")
parser.add_argument("--apply_flip", type=bool, default=False)
parser.add_argument("--apply_translation", type=bool, default=False)
parser.add_argument("--apply_rotation", type=bool, default=False)
parser.add_argument("--apply_zoom", type=bool, default=False)
parser.add_argument("--apply_brightness", type=bool, default=False)
parser.add_argument("--apply_contrast", type=bool, default=False)
parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
args = parser.parse_args()


if __name__ == "__main__":

    # Set random seeds for reproducibility of weights initialization
    # All three of these must be set in order to make the weights initialization reproducible
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(project=args.project, name=timestr, config=vars(args))
    
    label_dict_map = {'A': 0, 'B': 1, 'S': 2, 'P': 3}

    X_train = glob.glob(args.data_path + '/train/**/*.png', recursive=True)
    y_train = [os.path.basename(os.path.dirname(f)) for f in X_train]
    y_train = [label_dict_map[l] for l in y_train]

    X_val = glob.glob(args.data_path + '/val/**/*.png', recursive=True)
    y_val = [os.path.basename(os.path.dirname(f)) for f in X_val]
    y_val = [label_dict_map[l] for l in y_val]


    def map_fn(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, args.image_size)
        return image, label


    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.map(map_fn)
    ds_train = ds_train.shuffle(len(X_train)).batch(args.batch_size)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.map(map_fn)
    ds_val = ds_val.batch(args.batch_size)    
    
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, pooling='avg')
    wandb.config.update({'base_model': base_model.name})
    
    layers = []
    
    if args.apply_flip:
        layers.append(tf.keras.layers.RandomFlip())
    if args.apply_translation:
        layers.append(tf.keras.layers.RandomTranslation(0.2, 0.2))
    if args.apply_rotation:
        layers.append(tf.keras.layers.RandomRotation(0.2))
    if args.apply_zoom:
        layers.append(tf.keras.layers.RandomZoom(0.2))
    if args.apply_brightness:
        layers.append(tf.keras.layers.RandomBrightness(0.1))
    if args.apply_contrast:
        layers.append(tf.keras.layers.RandomContrast(0.5))
    
    layers.append(base_model)
    layers.append(tf.keras.layers.Dense(4, activation='softmax'))
    
    model = tf.keras.Sequential(layers)
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate_start,
        end_learning_rate=args.learning_rate_end,
        decay_steps=args.epochs*len(ds_train),
        power=2.0,
        )        
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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
    
    os.makedirs('./models', exist_ok=True)
    model.save(f'./models/{timestr}.keras')
    