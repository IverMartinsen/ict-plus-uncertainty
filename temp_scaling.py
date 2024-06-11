import numpy as np
import tensorflow as tf
from pSGLD_v2 import pSGLangevinDynamics
from utils import make_dataset, load_data

image_size = [224, 224]
batch_size = 64
data_path = "./data/Training_Dataset_Cropped_Split/"

X_train, y_train = load_data(data_path + '/train/')
X_val, y_val = load_data(data_path + '/val/')

ds_train = make_dataset(X_train, y_train, image_size, batch_size, shuffle=True)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False)






path_to_model = './SGLD/20240605_132312/100.keras'

model = tf.keras.models.load_model(path_to_model, custom_objects={'pSGLangevinDynamics': pSGLangevinDynamics})

# get the logits
model_logits = tf.keras.Model(inputs=model.input, outputs=model.layers[5].output)
model_logits.summary()



class TempScaling(tf.keras.layers.Layer):
    def __init__(self):
        super(TempScaling, self).__init__()
        self.scaling_parameter = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        return inputs * self.scaling_parameter

    def get_config(self):
        return {'scaling_parameter': self.scaling_parameter}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model_logits = tf.keras.Sequential([
    tf.keras.applications.EfficientNetB0(include_top=False, weights=None, pooling='avg'),
    tf.keras.layers.Dense(4, activation='linear'),
    TempScaling(),
    tf.keras.layers.Activation('softmax'),
])

model_logits.layers[0].set_weights(model.layers[6].get_weights())
model_logits.layers[1].set_weights(model.layers[7].get_weights())
model_logits.layers[0].trainable = False
model_logits.layers[1].trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)


model_logits.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_logits.fit(
    ds_val, 
    epochs=10, 
    validation_data=ds_val,
    )

model_logits.layers[-2].get_weights()

model.save('./SGLD/20240605_132312/100_temp_scaled.keras')