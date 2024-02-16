import os
import tensorflow as tf


data_path = './data/Training_Dataset/'
filenames = os.listdir(data_path)
labels = [f[0] for f in filenames]
image_size = (224, 224)


ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
ds = ds.map(lambda x, y: (tf.io.read_file(data_path + x), y))
ds = ds.map(lambda x, y: (tf.image.decode_png(x, channels=3), y))
ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
ds = ds.map(lambda x, y: (x / 255., y))


for data, label in ds.take(10):
    print(data, label)

import matplotlib.pyplot as plt

plt.imshow(data)
plt.title(label)
plt.show()