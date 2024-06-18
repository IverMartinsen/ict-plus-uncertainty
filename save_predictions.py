import os
import glob
import json
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from utils import (
    lab_to_int, 
    lab_to_long, 
    make_dataset, 
    load_data, 
    store_predictions,
    store_confusion_matrix,
    store_summary_stats,
)
from optimizer import StochasticGradientLangevinDynamics
from schedule import PolynomialDecay


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_json", type=str, default='./models/ensemble.json')
parser.add_argument("--path_to_models", type=str, default='./models/')
parser.add_argument("--from_folder", type=bool, default=False)
parser.add_argument("--image_size", type=int, nargs="+", default=[224, 224])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--destination", type=str, default='stats/ensemble_stats')
parser.add_argument("--path_to_val_data", type=str, default='./data/Man vs machine_Iver_cropped/')
args = parser.parse_args()


if __name__ == '__main__':

    print('Loading models...')
    if args.from_folder:
        models = glob.glob(args.path_to_models + '*.keras')
        models.sort()
    else:
        config = json.load(open(args.path_to_json, 'r'))
        keys = [k for k in config.keys() if 'model' in k]
        models = [str(config[k]) for k in keys]
        models = [os.path.join(args.path_to_models, m) for m in models]
    
    custom_objects = {
        'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
        'PolynomialDecay': PolynomialDecay,
        }
    models = [tf.keras.models.load_model(m, custom_objects=custom_objects) for m in models]

    assert len(models) > 0, 'No models found'

    os.makedirs(args.destination, exist_ok=True)

    print('Loading validation data...')
    X_val, y_val = load_data(args.path_to_val_data)
    assert len(X_val) > 0, 'No validation data found'
    ds_val = make_dataset(X_val, y_val, args.image_size, args.batch_size, shuffle=False, seed=1)

    print('Making predictions...')
    Y_pred = np.empty((len(y_val), len(models), len(lab_to_int)))
    for i, model in enumerate(models):
        predictions = model.predict(ds_val)
        Y_pred[:, i, :] = predictions

    print('Saving predictions...')
    np.save(os.path.join(args.destination, 'predictions.npy'), Y_pred)
    np.save(os.path.join(args.destination, 'labels.npy'), y_val)
    np.save(os.path.join(args.destination, 'filenames.npy'), X_val)
        
    print('Evaluation complete.')
    