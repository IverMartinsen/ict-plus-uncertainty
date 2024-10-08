import matplotlib.pyplot as plt
import tensorflow as tf
from utils.utils import make_dataset, load_data

data_path="./data/Training_Dataset_Cropped_Split/"
image_size = [224, 224]
batch_size = 64
random_seed = 1
s = 2.0

X_train, y_train = load_data(data_path + '/train/')

ds_train = make_dataset(X_train, y_train, image_size, batch_size, shuffle=True, seed=random_seed)

layers = []
layers.append(tf.keras.layers.RandomFlip(seed=random_seed))
layers.append(tf.keras.layers.RandomRotation(0.1*s, seed=random_seed))
layers.append(tf.keras.layers.RandomTranslation(0.1*s, 0.1*s, seed=random_seed))
layers.append(tf.keras.layers.RandomZoom((-0.1*s, 0.0), seed=random_seed))
layers.append(tf.keras.layers.RandomContrast(0.2*s, seed=random_seed))

augmentation = tf.keras.Sequential(layers)

fig, axs = plt.subplots(4, 2, figsize=(10, 20))

for batch in ds_train:
    X_batch, y_batch = batch
    images = augmentation(X_batch, training=True)
    for i in range(4):
        axs[i, 0].imshow(X_batch[i] / 255.)
        axs[i, 0].set_title(f"Original", fontsize=20)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(images[i] / 255.)
        axs[i, 1].set_title(f"Augmented", fontsize=20)
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'augmentations_strength_{s}.png', bbox_inches='tight', dpi=300)
    plt.close()
    break
