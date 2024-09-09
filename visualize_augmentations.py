import matplotlib.pyplot as plt
import tensorflow as tf
from utils import make_dataset, load_data

data_path="./data/Training_Dataset_Cropped_Split/"
image_size = [224, 224]
batch_size = 64
random_seed = 1
path_to_model = '/Users/ima029/Desktop/IKT+ Uncertainty/Repository/tta/20240610_135338.keras'

X_train, y_train = load_data(data_path + '/train/')
X_val, y_val = load_data(data_path + '/val/')

ds_train = make_dataset(X_train, y_train, image_size, batch_size, shuffle=True, seed=random_seed)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False, seed=random_seed)

layers = []
layers.append(tf.keras.layers.RandomFlip(seed=random_seed))
layers.append(tf.keras.layers.RandomRotation(0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomTranslation(0.2, 0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomZoom((-0.2, 0.0), seed=random_seed))
layers.append(tf.keras.layers.RandomBrightness(0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomContrast(0.4, seed=random_seed))

augmentation = tf.keras.Sequential(layers)


for batch in ds_train:
    X_batch, y_batch = batch
    images = augmentation(X_batch, training=True)
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        axs[0, i].imshow(X_batch[i] / 255.)
        axs[0, i].set_title(f"Original {i + 1}")
        axs[0, i].axis('off')
        axs[1, i].imshow(images[i] / 255.)
        axs[1, i].set_title(f"Augmented {i + 1}")
        axs[1, i].axis('off')
        axs[2, i].imshow(X_batch[i + 4] / 255.)
        axs[2, i].set_title(f"Original {i + 5}")
        axs[2, i].axis('off')
        axs[3, i].imshow(images[i + 4] / 255.)
        axs[3, i].set_title(f"Augmented {i + 5}")
        axs[3, i].axis('off')
    plt.show()
    break
