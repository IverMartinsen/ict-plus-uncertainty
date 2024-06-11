import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
from optimizer import StochasticGradientLangevinDynamics
from utils import (
    make_dataset, 
    load_data, 
    make_calibration_plots, 
    make_ordered_calibration_plot, 
    lab_to_long, 
    plot_images, 
    int_to_lab
)

# hyperparameters
image_size = [224, 224]
batch_size = 32
destination = 'confidence_stats'
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'
path_to_model = './ensemble/20240606_122403.keras'

os.makedirs(destination, exist_ok=True)

# load the models
model = tf.keras.models.load_model(path_to_model, custom_objects={'pSGLangevinDynamics': StochasticGradientLangevinDynamics})

# load the validation data
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False)
y_val = np.array(y_val)

# get the predictions
Y_pred = model.predict(ds_val)
y_pred = Y_pred.argmax(axis=1)
confidence = np.max(Y_pred, axis=1)
variance = Y_pred.max(axis=1) * (1 - Y_pred.max(axis=1))
total_variance = (Y_pred * (1 - Y_pred)).sum(axis=1)
loss = -np.log(Y_pred[np.arange(len(y_val)), y_val])
x = (1 - confidence).reshape(-1, 1)
y = Y_pred.argmax(axis=1) == y_val

# store the predictions
columns = ['filename', 'label', 'pred', 'confidence', 'loss', 'variance', 'total_variance']
df = pd.DataFrame(columns=columns)

df['filename'] = X_val
df['label'] = y_val
df['pred'] = y_pred
df['loss'] = loss
df['confidence'] = confidence
df['variance'] = variance
df['total_variance'] = total_variance
df.to_csv(os.path.join(destination, 'ensemble_predictions.csv'), index=False)






total_wrong = np.array(y_val != y_pred).sum()
percentage = 0.75
threshold = np.quantile(x, percentage)
score = int(np.array((y_val != y_pred)[x.flatten() < threshold]).sum())

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, np.log(loss), c=y_val == Y_pred.argmax(axis=1))
ax.axvline(x=np.quantile(x, percentage), color='black', linestyle='--')
ax.set_ylabel('Loss')
ax.set_xlabel(r'1 - $\hat{y}$')
ax.set_title(f'Confidence vs Loss ({score}/{total_wrong} below {percentage} quantile)', fontsize=10, fontweight='bold')
ax.legend(['correct', f'{percentage} quantile'])
plt.savefig(os.path.join(destination, 'confidence_vs_loss.png'), bbox_inches='tight', dpi=300)


make_calibration_plots(Y_pred, y_val, destination, num_bins=9)
make_ordered_calibration_plot(Y_pred, y_val, destination, num_bins=20)

# Plot the most confident and least confident images
most_confident = np.argsort(confidence)[-25:]
filenames = np.array(X_val)[most_confident]
labels = y_val[most_confident]
labels = [lab_to_long[int_to_lab[l]] for l in labels]
preds = y_pred[most_confident]
preds = [lab_to_long[int_to_lab[p]] for p in preds]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(Image.open(filenames[i]).resize(image_size))
    ax.set_title(f'Label: {labels[i]}\nPred: {preds[i]}\nFile: {os.path.basename(filenames[i])}', fontsize=6, fontweight='bold')
    ax.axis('off')
# increase the space between the plots
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(destination, 'most_confident.png'), bbox_inches='tight', dpi=300)

least_confident = np.argsort(confidence)[:25]
filenames = np.array(X_val)[least_confident]
labels = y_val[least_confident]
labels = [lab_to_long[int_to_lab[l]] for l in labels]
preds = y_pred[least_confident]
preds = [lab_to_long[int_to_lab[p]] for p in preds]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(Image.open(filenames[i]).resize(image_size))
    ax.set_title(f'Label: {labels[i]}\nPred: {preds[i]}\nFile: {os.path.basename(filenames[i])}', fontsize=6, fontweight='bold')
    ax.axis('off')
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(destination, 'least_confident.png'), bbox_inches='tight', dpi=300)

# Store total accuracy
summary= pd.DataFrame(columns=['accuracy', 'loss'], index=['model'])
summary['accuracy'] = (y_val == y_pred).mean()
summary['loss'] = -np.log(Y_pred[np.arange(len(y_val)), y_val]).mean()
summary.to_csv(os.path.join(destination, 'summary.csv'))

# Store class wise accuracy
class_wise_accuracy = classification_report(y_val, y_pred, target_names=list(lab_to_long.values()), output_dict=True)
class_wise_df = pd.DataFrame(class_wise_accuracy).T
class_wise_df.to_csv(os.path.join(destination, 'class_wise_accuracy.csv'))
