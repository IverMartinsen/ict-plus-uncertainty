import numpy as np
import tensorflow as tf


class StochasticGradientLangevinDynamics(tf.keras.optimizers.Optimizer):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics.
    Compatability: Tensorflow 2.9
    Implementation is without the gamma term.
    """
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        epsilon=1e-7,
        data_size=1,
        burnin=0,
        weight_decay=None,
        name="pSGLD",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._rho = rho # alpha parameter in preconditioner decay rate (alpha)
        self._epsilon = epsilon # diagonal bias (lambda)
        self._data_size = tf.convert_to_tensor(data_size, name='data_size')
        self._burnin = tf.convert_to_tensor(burnin, name='burnin')
        self._weight_decay = weight_decay
    

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "velocity")

    def _resource_apply_dense(self, gradient, variable):
        
        # Assumes reverse sign of gradient is passed
        gradient = -gradient
        
        #lr = tf.cast(self.learning_rate, variable.dtype)
        lr = tf.identity(self._decayed_lr(var_dtype=variable.dtype))
        rho = self._rho
        velocity = self.get_slot(variable, "velocity")
        n = tf.cast(self._data_size, gradient.dtype)
        
        # V(theta_t), same as in RMSprop
        v = velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient)) # V(theta_t)
        
        denominator = v + self._epsilon # V(theta_t) + lambda
        # G(theta_t) = 1/sqrt(V(theta_t) + lambda), same as in RMSprop
        preconditioner = tf.math.rsqrt(denominator) # G(theta_t)
        # Differ from RMSprop by scaling with data size * 0.5
        mean = preconditioner * gradient
        if self._weight_decay is not None:
            mean += preconditioner * self._weight_decay * tf.square(variable) / n
        mean *= lr
        stddev = tf.where(
            tf.squeeze(self.iterations > tf.cast(self._burnin, tf.int64)),
            tf.cast(tf.math.sqrt(lr), gradient.dtype),
            tf.zeros([], gradient.dtype)
            )
        # Instead of scaling up the mean, we scale down the stddev
        # This is to align the learning rate with normal SGD and Adam values
        stddev *= tf.sqrt(preconditioner)
        stddev *= 2.0
        stddev /= n
        
        result_shape = tf.broadcast_dynamic_shape(tf.shape(mean), tf.shape(stddev))
        noisy_grad = tf.random.normal(shape=result_shape, mean=mean, stddev=stddev, dtype=gradient.dtype)

        var_update = variable.assign_add(noisy_grad)
        updates = [var_update, velocity]
        
        return tf.group(*updates)

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                **config,
                "learning_rate": self._serialize_hyperparameter('learning_rate'),
                "decay": self._initial_decay,
                "rho": self._rho,
                "epsilon": self._epsilon,
                "data_size": self._data_size.numpy(),
                "burnin": self._burnin.numpy(),
                "weight_decay": self._weight_decay,
            }
        )
        return config
