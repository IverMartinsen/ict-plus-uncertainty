import os
import glob
import wandb
import argparse
import tensorflow as tf
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--data_path", type=str, default="./data/Training_Dataset_Cropped_Split/")
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--project", type=str, default="ict-plus-uncertainty")
args = parser.parse_args()


if __name__ == "__main__":

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


    def augment(image, label):
        image /= 255.0 # normalize to [0,1] range
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0, 1)
        image *= 255.0 # back to [0,255] range
        return image, label


    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.map(map_fn)
    ds_train = ds_train.map(augment)
    ds_train = ds_train.shuffle(len(X_train)).batch(args.batch_size)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.map(map_fn)
    ds_val = ds_val.batch(args.batch_size)    
    
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, pooling='avg')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(4, activation='softmax')
    ])    
    
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        ds_train, 
        epochs=args.epochs, 
        validation_data=ds_val, 
        callbacks=[wandb.keras.WandbCallback(save_model=False)],
        )
    