import tensorflow as tf 

class GlobalResponseNormalization (tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma",
                                     shape=(input_shape[-1],),
                                     initializer="ones",
                                     trainable=True
                                     )
        self.beta = self.add_weight(name="beta",
                            shape=(input_shape[-1],),
                            initializer="zeros",
                            trainable=True)
    
    def call(self, inputs):
        # L2 norm of each channel
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=[1, 2], keepdims=True) + 1e-12)
        # Normalize
        inputs_normalized = inputs / l2_norm
        # Scale and shift
        return self.gamma * inputs_normalized + self.beta

# # Example usage in a model
# input_tensor = tf.keras.Input(shape=(256, 256, 32))
# x = GlobalResponseNormalization()(input_tensor)
# model = tf.keras.Model(inputs=input_tensor, outputs=x)
# model.summary()