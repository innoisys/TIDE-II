import tensorflow as tf
import tensorflow.keras.layers as layers


class TopLayer(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

        self.conv_1x1 = layers.Conv2D(self.filters, (1, 1), activation='relu', strides=1, padding="same",
                                      name="top_layer_1x1")
        self.conv_2x2 = layers.Conv2D(self.filters//3, (2, 2), activation='relu', strides=1, padding="same",
                                      name="top_layer_2x2")
        self.conv_4x4 = layers.Conv2D(self.filters//3, (4, 4), activation='relu', strides=1, padding="same",
                                      name="top_layer_4x4")
        self.conv_8x8 = layers.Conv2D(self.filters//3, (8, 8), activation='relu', strides=1, padding="same",
                                      name="top_layer_8x8")

        self.concat = layers.Concatenate(axis=-1)
        self.point_wise_conv = layers.Conv2D(self.filters, (1, 1), 1, activation=None, use_bias=False,
                                             padding='same', name="top_layer_point_wise")
        self.feat_fusion = layers.Conv2D(self.filters, (1, 1), 1, activation=None, use_bias=False,
                                         padding='same', name="top_layer_fusion")

        self.addition = layers.Add()
        self.gelu = layers.Activation('gelu')
        self.final_conv = layers.Conv2D(self.filters, (1, 1),  activation='relu', strides=1, padding="same",
                                        name="top_layer_out")

    def call(self, inputs, training=False):
        x = self.conv_1x1(inputs, training=training)

        feats_2x2 = self.conv_2x2(x, training=training)
        feats_4x4 = self.conv_4x4(x, training=training)
        feats_8x8 = self.conv_8x8(x, training=training)

        concatenated = self.concat([feats_2x2, feats_4x4, feats_8x8])
        concatenated = self.point_wise_conv(concatenated)

        concatenated = self.feat_fusion(concatenated)
        x = self.addition([inputs, concatenated])
        x = self.gelu(x)
        x = self.final_conv(x)
        return x


class Sampling(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
