import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from utils.utils import make_dataset, load_data
from utils.optimizer import StochasticGradientLangevinDynamics
from utils.schedule import PolynomialDecay

parser = ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/tta_results_s1/')
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=30)
parser.add_argument("--s", type=float, default=0.1)
parser.add_argument("--data_path", type=str, default="./data/Man vs machine_Iver_cropped/")
parser.add_argument("--path_to_model", type=str, default='./models/20240819_144329.keras')
args = parser.parse_args()

if __name__ == '__main__':

    os.makedirs(args.destination, exist_ok=True)

    X_val, y_val = load_data(args.data_path)
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=args.random_seed)

    model = tf.keras.models.load_model(args.path_to_model, custom_objects={
        'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
        'PolynomialDecay': PolynomialDecay
        })

    layers = []
    layers.append(tf.keras.layers.RandomFlip(seed=args.random_seed))
    layers.append(tf.keras.layers.RandomRotation(args.s, seed=args.random_seed))
    layers.append(tf.keras.layers.RandomTranslation(args.s, args.s, seed=args.random_seed))
    layers.append(tf.keras.layers.RandomZoom((-args.s, 0.0), seed=args.random_seed))
    layers.append(tf.keras.layers.RandomContrast(2*args.s, seed=args.random_seed))

    augmentation = tf.keras.Sequential(layers)

    y_pred = np.empty((len(y_val), args.num_samples, 4))
    labs = np.concatenate([y for _, y in ds_val], axis=0)

    for i in range(args.num_samples):
        print(f"================ Sample {i + 1} ================")
        preds = []
        for batch in ds_val:
            print(f"Batch {len(preds) + 1}/{len(ds_val)}")
            X_batch, y_batch = batch
            images = augmentation(X_batch, training=True)
            preds.append(model(images, training=False))
        preds = np.concatenate(preds, axis=0)
        y_pred[:, i, :] = preds

    np.save(os.path.join(args.destination, 'predictions.npy'), y_pred)
    np.save(os.path.join(args.destination, 'labels.npy'), labs)
    np.save(os.path.join(args.destination, 'filenames.npy'), X_val)
