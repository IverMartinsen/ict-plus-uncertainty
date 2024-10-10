import os
import argparse
import numpy as np
import tensorflow as tf
from utils.utils import lab_to_int, make_dataset, load_data
from utils.optimizer import StochasticGradientLangevinDynamics
from utils.schedule import PolynomialDecay


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='./models/20240819_144329.keras')
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=30)
parser.add_argument("--augmentation_strength", type=float, default=0.1)
parser.add_argument("--destination", type=str, default='./results/tta_results_s1/')
parser.add_argument("--data_path", type=str, default="./data/Man vs machine_Iver_cropped/")
args = parser.parse_args()


if __name__ == '__main__':

    os.makedirs(args.destination, exist_ok=True)

    print('Loading validation data...')
    X_val, y_val = load_data(args.data_path)
    assert len(X_val) > 0, 'No validation data found'
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=args.random_seed)

    print('Loading models...')
    
    custom_objects = {
        'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
        'PolynomialDecay': PolynomialDecay
        }
    model = tf.keras.models.load_model(args.path, custom_objects=custom_objects)
    
    print('Making predictions...')

    layers = []
    layers.append(tf.keras.layers.RandomFlip(seed=args.random_seed))
    layers.append(tf.keras.layers.RandomRotation(args.augmentation_strength, seed=args.random_seed))
    layers.append(tf.keras.layers.RandomTranslation(args.augmentation_strength, args.augmentation_strength, seed=args.random_seed))
    layers.append(tf.keras.layers.RandomZoom((-args.augmentation_strength, 0.0), seed=args.random_seed))
    layers.append(tf.keras.layers.RandomContrast(2*args.augmentation_strength, seed=args.random_seed))

    augmentation = tf.keras.Sequential(layers)

    Y_pred = np.empty((len(y_val), args.num_samples, len(lab_to_int)))
    for i in range(args.num_samples):
        print(f"================ Sample {i + 1} ================")
        preds = []
        for batch in ds_val:
            print(f"Batch {len(preds) + 1}/{len(ds_val)}")
            X_batch, y_batch = batch
            images = augmentation(X_batch, training=True)
            preds.append(model(images, training=False))
        preds = np.concatenate(preds, axis=0)
        Y_pred[:, i, :] = preds

    print('Saving predictions...')
    np.save(os.path.join(args.destination, 'predictions.npy'), Y_pred)
    np.save(os.path.join(args.destination, 'labels.npy'), y_val)
    np.save(os.path.join(args.destination, 'filenames.npy'), X_val)
    
    print('Evaluation complete.')
    