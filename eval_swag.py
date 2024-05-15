import tensorflow as tf
from swag_utils import assign_weights, read_weights
from utils import make_dataset, load_data

path_to_weights = "./ensemble/20240312_155425.keras"

model = tf.keras.models.load_model(path_to_weights)

mean_weights = read_weights("swag/swag_diagonal_mean.npz")
mean_squared = read_weights("swag/swag_diagonal_squared_sum.npz")

variances = [s - m**2 for s, m in zip(mean_squared, mean_weights)]
variances = [tf.clip_by_value(v, 0, 1e5) for v in variances]



image_size = [224, 224]
batch_size = 32
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size)

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)



for i in range(10):
    delta = [tf.random.normal(shape=w.shape, stddev=tf.sqrt(w)) for w in variances]
    theta = [m + d for m, d in zip(mean_weights, delta)]
    assign_weights(model, theta)
    model.evaluate(ds_val)




for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(layer.trainable)