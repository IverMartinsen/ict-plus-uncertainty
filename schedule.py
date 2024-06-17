# ======================================================================================================================
# Description: This file contains the implementation of the PolynomialDecay learning rate schedule.
# This file is mostly copied from:
#   https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/schedules/learning_rate_schedule.py#L304-L457
# ======================================================================================================================

import tensorflow as tf


class PolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a polynomial decay schedule.

    It is commonly observed that a monotonically decreasing learning rate, whose
    degree of change is carefully chosen, results in a better performing model.
    This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

    It requires a `step` value to compute the decayed learning rate. You
    can just pass a TensorFlow variable that you increment at each training
    step.

    The schedule is a 1-arg callable that produces a decayed learning rate
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
            ) + end_learning_rate
    ```

    If `cycle` is True then a multiple of `decay_steps` is used, the first one
    that is bigger than `step`.

    ```python
    def decayed_learning_rate(step):
    decay_steps = decay_steps * ceil(step / decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
            ) + end_learning_rate
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.
    Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
    sqrt (i.e. power=0.5):

    ```python
    ...
    starter_learning_rate = 0.1
    end_learning_rate = 0.01
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    model.compile(optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_fn),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        warmup_steps=None,
        cycle=False,
        name=None):
        """Applies a polynomial decay to the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
            decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
            end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
            power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
            warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            The number of warmup steps that will be used to increase the learning rate.
            cycle: A boolean, whether or not it should cycle beyond decay_steps.
            name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
        """
        super(PolynomialDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "PolynomialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            end_learning_rate = tf.cast(self.end_learning_rate, dtype)
            power = tf.cast(self.power, dtype)

            global_step_recomp = tf.cast(step, dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype)
            # Add warmup_steps
            if self.warmup_steps:
                global_step_recomp = tf.maximum(global_step_recomp - self.warmup_steps, 0)
                decay_steps_recomp = tf.maximum(decay_steps_recomp - self.warmup_steps, 1)
            
            if self.cycle:
            # Find the first multiple of decay_steps that is bigger than
            # global_step. If global_step is zero set the multiplier to 1
                multiplier = tf.where(
                    tf.equal(global_step_recomp, 0), 1.0,
                    tf.math.ceil(global_step_recomp / self.decay_steps))
                decay_steps_recomp = tf.multiply(decay_steps_recomp, multiplier)
            else:
            # Make sure that the global_step used is not bigger than decay_steps.
                global_step_recomp = tf.minimum(global_step_recomp,
                                                        decay_steps_recomp)

            p = tf.divide(global_step_recomp, decay_steps_recomp)
            
            return tf.cond(tf.less(step, self.warmup_steps),
                    lambda: tf.add(
                        tf.multiply(initial_learning_rate - 0,
                                    step / self.warmup_steps),
                        0,
                        name=name),
                    lambda: tf.add(
                        tf.multiply(initial_learning_rate - end_learning_rate,
                                    tf.pow(1 - p, power)),
                        end_learning_rate,
                        name=name)
                    )
            
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name,
            "warmup_steps": self.warmup_steps
        }
        