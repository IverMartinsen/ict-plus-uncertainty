import os
import wandb
import argparse
import random
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import make_dataset, load_data
from swag_utils import SWAGDiagonalCallback


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--data_path", type=str, default="./data/Training_Dataset_Cropped_Split/")
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--project", type=str, default="ict-plus-uncertainty")
parser.add_argument("--save_path", type=str, default="./models/")
parser.add_argument("--apply_flip", type=bool, default=False)
parser.add_argument("--apply_translation", type=bool, default=False)
parser.add_argument("--apply_rotation", type=bool, default=False)
parser.add_argument("--apply_zoom", type=bool, default=False)
parser.add_argument("--apply_brightness", type=bool, default=False)
parser.add_argument("--apply_contrast", type=bool, default=False)
parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--path_to_weights", type=str, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
args = parser.parse_args()


if __name__ == "__main__":

    # Set random seeds for reproducibility of weights initialization
    # All three of these must be set in order to make the weights initialization reproducible
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(project=args.project, name=timestr, config=vars(args))

    X_train, y_train = load_data(args.data_path + '/train/')
    X_val, y_val = load_data(args.data_path + '/val/')
    
    ds_train = make_dataset(X_train, y_train, args.image_size, args.batch_size, shuffle=True)
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False)
    
    model = tf.keras.models.load_model(args.path_to_weights)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    destination = os.path.join(args.save_path, timestr)
    
    os.makedirs(destination, exist_ok=True)
    
    with open(os.path.join(destination, 'args.txt'), 'w') as f:
        f.write(str(vars(args)))
    
    model.fit(
        ds_train, 
        epochs=args.epochs, 
        validation_data=ds_val, 
        callbacks=[wandb.keras.WandbCallback(save_model=False), SWAGDiagonalCallback(model, destination)]
        )
    