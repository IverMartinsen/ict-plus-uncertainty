import os
import glob
import json
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


# hyperparameters
path_to_json = './models/ensemble.json'
path_to_models = './models/'
from_folder = False
image_size = [224, 224]
batch_size = 32
destination = 'stats/ensemble_stats'
path_to_val_data = './data/Man vs machine_Iver_cropped/'

if __name__ == '__main__':

    # load the models
    if from_folder:
        models = glob.glob(path_to_models + '*.keras')
        models.sort()
    else:
        config = json.load(open(path_to_json, 'r'))
        keys = [k for k in config.keys() if 'model' in k]
        models = [str(config[k]) for k in keys]
        models = [os.path.join(path_to_models, m) for m in models]
        models = [tf.keras.models.load_model(m, custom_objects={'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics}) for m in models]

    assert len(models) > 0, 'No models found'

    os.makedirs(destination, exist_ok=True)

    # load the validation data
    X_val, y_val = load_data(path_to_val_data)
    assert len(X_val) > 0, 'No validation data found'
    ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False, seed=1)

    # get the predictions
    Y_pred = np.empty((len(y_val), len(models), len(lab_to_int)))
    for i, model in enumerate(models):
        predictions = model.predict(ds_val)
        Y_pred[:, i, :] = predictions

    np.save(os.path.join(destination, 'predictions.npy'), Y_pred)
    
    # =============================================================================
    # STATISTICS
    # =============================================================================

    df = store_predictions(Y_pred, y_val, X_val, destination)

    store_confusion_matrix(df['label'], df['pred_mean'], destination)

    store_summary_stats(df, destination)

    class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
    class_wise_df = pd.DataFrame(class_wise_accuracy).T
    class_wise_df.to_csv(os.path.join(destination, 'class_wise_accuracy.csv'))
