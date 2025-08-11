import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow.keras import backend


class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


class StochasticDepth(layers.Layer):
    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class ConvNeXtBlock(layers.Layer):
    def __init__(self, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name_prefix=None):
        super().__init__(name=name_prefix or f"prestem{backend.get_uid('prestem')}")
        self.depthwise_conv = layers.Conv2D(
            filters=projection_dim, kernel_size=7, padding="same", groups=projection_dim,
            name=self.name + "_depthwise_conv"
        )
        self.pointwise_conv1 = layers.Dense(4 * projection_dim, name=self.name + "_pointwise_conv_1")
        self.act = layers.Activation("gelu", name=self.name + "_gelu")
        self.pointwise_conv2 = layers.Dense(projection_dim, name=self.name + "_pointwise_conv_2")
        self.layer_scale = LayerScale(layer_scale_init_value, projection_dim, name=self.name + "_layer_scale") \
            if layer_scale_init_value is not None else None
        self.stochastic_depth = StochasticDepth(drop_path_rate, name=self.name + "_stochastic_depth") \
            if drop_path_rate else layers.Activation("linear", name=self.name + "_identity")

    def call(self, inputs, training=False):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        if self.layer_scale:
            x = self.layer_scale(x)
        x = self.stochastic_depth(x, training=training)
        return inputs + x


class ConvNeXtBlockTransposed(layers.Layer):
    def __init__(self, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name_prefix=None):
        super().__init__(name=name_prefix or f"poststem{backend.get_uid('poststem')}")
        self.projection_dim = projection_dim
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value

        self.depthwise_conv_trans = layers.Conv2DTranspose(
            filters=projection_dim, kernel_size=7, padding="same",
            groups=projection_dim, name=self.name + "_depthwise_conv_trans"
        )
        self.pointwise_conv1 = layers.Dense(4 * projection_dim, name=self.name + "_pointwise_conv_1")
        self.act = layers.Activation("gelu", name=self.name + "_gelu")
        self.pointwise_conv2 = layers.Dense(projection_dim, name=self.name + "_pointwise_conv_2")

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale(layer_scale_init_value, projection_dim, name=self.name + "_layer_scale")
        else:
            self.layer_scale = None

        if drop_path_rate:
            self.stochastic_depth = StochasticDepth(drop_path_rate, name=self.name + "_stochastic_depth")
        else:
            self.stochastic_depth = layers.Activation("linear", name=self.name + "_identity")

    def call(self, inputs, training=False):
        x = self.depthwise_conv_trans(inputs)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        if self.layer_scale:
            x = self.layer_scale(x)
        x = self.stochastic_depth(x, training=training)
        return inputs + x


