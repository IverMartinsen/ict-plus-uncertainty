import os
import wandb
import argparse
import random
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import make_dataset, load_data
from swag_utils import SWAGDiagonalCallback
from optimizer import StochasticGradientLangevinDynamics

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--data_path", type=str, default="./data/Training_Dataset_Cropped_Split/")
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--project", type=str, default="ict-plus-uncertainty")
parser.add_argument("--save_path", type=str, default="./models/")
parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--path_to_weights", type=str, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--burnin", type=int, default=1e4)
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

    ds_train = make_dataset(X_train, y_train, args.image_size, args.batch_size, shuffle=True, seed=args.random_seed)
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=args.random_seed)

    model = tf.keras.models.load_model(args.path_to_weights, custom_objects={'pSGLangevinDynamics': StochasticGradientLangevinDynamics})

    sample_size = len(X_train)
    steps_per_epoch = np.ceil(sample_size / args.batch_size).astype(int)
    burnin = steps_per_epoch*100 # number of training steps before starting to add gradient noise

    # update optimizer hyperparameters
    model.optimizer.learning_rate = args.learning_rate
    model.optimizer._burnin = tf.convert_to_tensor(0, name='burnin')
    
    destination = os.path.join(args.save_path, timestr)

    os.makedirs(destination, exist_ok=True)

    with open(os.path.join(destination, 'args.txt'), 'w') as f:
        f.write(str(vars(args)))

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(destination, "{epoch:02d}.keras"),
        save_best_only=False,
        save_weights_only=False,
        )

    wandb_callback = wandb.keras.WandbCallback(save_model=False)

    lr_variable = wandb.define_metric('learning_rate')

    def log_lr(epoch, logs):
        wandb.log({'learning_rate': model.optimizer._decayed_lr(tf.float32).numpy()}, commit=False)

    lr_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_lr)

    callbacks = [
        model_checkpoint,
        wandb_callback,
        lr_callback,
        SWAGDiagonalCallback(model, destination),
        ]

    model.fit(
        ds_train, 
        epochs=args.epochs, 
        validation_data=ds_val, 
        callbacks=callbacks,
        )