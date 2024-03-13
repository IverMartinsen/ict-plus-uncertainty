import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# hyperparameters
image_size = [224, 224]
batch_size = 32

# load the models
path_to_models = './ensemble/'
models = glob.glob(path_to_models + '*.keras')
models.sort()
models = [tf.keras.models.load_model(m) for m in models]

# load the validation data
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'

lab_to_int = {'A': 0, 'B': 1, 'S': 2, 'P': 3}
int_to_lab = {v: k for k, v in lab_to_int.items()}
lab_to_long = {'A': 'Benthic agglutinated', 'B': 'Benthic calcareous', 'S': 'Sediment', 'P': 'Planktic'}

X_val = glob.glob(path_to_val_data + '**/*.png', recursive=True)
y_val = [os.path.basename(os.path.dirname(f)) for f in X_val]
y_val = [lab_to_int[l] for l in y_val]

def map_fn(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image, label

ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
ds_val = ds_val.map(map_fn)
ds_val = ds_val.batch(batch_size)

# get the predictions
Y_pred = np.empty((len(y_val), len(models), len(lab_to_int)))
for i, model in enumerate(models):
    model = tf.keras.models.load_model(model)
    predictions = model.predict(ds_val)
    Y_pred[:, i, :] = predictions

# create a dataframe to store the predictions
columns = ['filename', 'label'] + [f'model_{i}' for i in range(len(models))]
df = pd.DataFrame(columns=columns)

df['filename'] = X_val
df['label'] = y_val

for i in range(len(models)):
    df[f'model_{i}'] = Y_pred[:, i, :].argmax(axis=1)
    
df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
df['pred_mode'] = df.iloc[:, 2:].mode(axis=1)[0] # majority vote
df['pred_mean'] = Y_pred.mean(axis=1).argmax(axis=1) # mean prediction
df['loss'] = -np.log(Y_pred.mean(axis=1)[np.arange(len(y_val)), y_val])

# save the predictions
df.to_csv('ensemble_predictions.csv', index=False)

# group the data
group1 = df[(df['agree'] == True) & (df['correct'] == True)]
group2 = df[(df['agree'] == False) & (df['correct'] == True)]
group3 = df[(df['agree'] == True) & (df['correct'] == False)]
group4 = df[(df['agree'] == False) & (df['correct'] == False)]

# plot tricky images
fig, axs = plt.subplots(1, 5, figsize=(10, 10))

for i, ax in enumerate(axs):
    filename = group3.iloc[i, 0]
    label = lab_to_long[int_to_lab[group3.iloc[i, 1]]]
    pred = lab_to_long[int_to_lab[group3.iloc[i, -1]]]
    ax.imshow(Image.open(filename).resize(image_size))
    ax.set_title(f'Label: {label}\nPred: {pred}', fontsize=6, fontweight='bold')
    ax.axis('off')
plt.savefig('agree_wrong.png', bbox_inches='tight', dpi=300)

# plot difficult images
fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axs.flatten()):
    try:
        filename = group4.iloc[i, 0]
        label = lab_to_long[int_to_lab[group4.iloc[i, 1]]]
        pred = lab_to_long[int_to_lab[group4.iloc[i, -1]]]
        ax.imshow(Image.open(filename).resize(image_size))
        ax.set_title(f'Label: {label}\nPred: {pred}', fontsize=6, fontweight='bold')
    except:
        pass
    ax.axis('off')
plt.savefig('disagree_wrong.png', bbox_inches='tight', dpi=300)

# confusion matrix
confusion = confusion_matrix(df['label'], df['pred_mean'])
disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)

# total accuracy
summary= pd.DataFrame(columns=[f'model_{i}' for i in range(len(models))] + ['majority_vote', 'mean_vote'], index=['accuracy'])
for i in range(len(models)):
    summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
summary['majority_vote'] = (df['label'] == df['pred']).mean()
summary['mean_vote'] = (df['label'] == df['pred_mean']).mean()
summary.to_csv('ensemble_summary.csv')

# class wise accuracy
class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
class_wise_df = pd.DataFrame(class_wise_accuracy).T
class_wise_df.to_csv('class_wise_accuracy.csv')

# epistemic uncertainty
ep_cov = np.matmul(Y_pred[:,:, :, np.newaxis], Y_pred[:, :, np.newaxis, :])
ep_cov[:, :, np.arange(len(lab_to_int)), np.arange(len(lab_to_int))] = (Y_pred * (1 - Y_pred))
ep_cov = ep_cov.mean(axis=1)
# aleatoric uncertainty
al_cov = np.zeros((len(y_val), len(lab_to_int), len(lab_to_int)))
for i in range(len(y_val)):
    al_cov[i] = np.cov(Y_pred[i].T)
# total uncertainty
tot_cov = ep_cov + al_cov
# summarize the uncertainty
gen_var = np.linalg.det(tot_cov)
tot_var = np.trace(tot_cov, axis1=1, axis2=2)

# plot the uncertainty
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.scatter(tot_var, df['loss'], c=df['correct'])
ax1.set_ylabel('Loss')
ax1.set_xlabel(r'tr$(\Sigma)$')
ax1.set_title('Total variance vs Loss', fontsize=10, fontweight='bold')
ax1.legend(['correct', 'wrong'])

ax2.scatter(np.log(gen_var), df['loss'], c=df['correct'])
ax2.set_ylabel('Loss')
ax2.set_xlabel(r'log|$\Sigma$|')
ax2.set_title('Generalized variance vs Loss', fontsize=10, fontweight='bold')
ax2.legend(['correct', 'wrong'])

plt.savefig('uncertainty.png', bbox_inches='tight', dpi=300)

# Group the data based on the uncertainty
threshold = np.median(tot_var)
group1 = df[(tot_var < threshold) & (df['pred_mean'] == df['label'])]
group2 = df[(tot_var < threshold) & (df['pred_mean'] != df['label'])]
group3 = df[(tot_var >= threshold) & (df['pred_mean'] == df['label'])]
group4 = df[(tot_var >= threshold) & (df['pred_mean'] != df['label'])]

# plot tricky images
fig, axs = plt.subplots(1, 2, figsize=(10, 10))

for i, ax in enumerate(axs):
    filename = group2.iloc[i, 0]
    label = lab_to_long[int_to_lab[group2.iloc[i, 1]]]
    pred = lab_to_long[int_to_lab[group2.iloc[i, -1]]]
    ax.imshow(Image.open(filename).resize(image_size))
    ax.set_title(f'Label: {label}\nPred: {pred}', fontsize=6, fontweight='bold')
    ax.axis('off')
plt.savefig('agree_wrong_tot_var.png', bbox_inches='tight', dpi=300)

# plot difficult images
fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axs.flatten()):
    try:
        filename = group4.iloc[i, 0]
        label = lab_to_long[int_to_lab[group4.iloc[i, 1]]]
        pred = lab_to_long[int_to_lab[group4.iloc[i, -1]]]
        ax.imshow(Image.open(filename).resize(image_size))
        ax.set_title(f'Label: {label}\nPred: {pred}', fontsize=6, fontweight='bold')
    except:
        pass
    ax.axis('off')
plt.savefig('disagree_wrong_tot_var.png', bbox_inches='tight', dpi=300)

