import numpy as np
import tensorflow.keras.layers as layers

from tensorflow.keras import Model
from tensorflow.keras import Sequential

from model.tidev2_utils import TopLayer, Sampling
from model.convnext_modules import ConvNeXtBlock, ConvNeXtBlockTransposed


class ConvNeXtEncoderTiny(Model):
    def __init__(self,
                 depths=[3, 3, 9, 3],
                 projection_dims=[96, 192, 384, 768],
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 model_name="convnext",
                 latent_dim=None):
        super().__init__(name=model_name)
        self.latent_dim = latent_dim
        self.depths = depths
        self.projection_dims = projection_dims

        # Stem
        self.stem = Sequential([
            layers.Conv2D(projection_dims[0], kernel_size=4, strides=4, name=model_name + "_stem_conv"),
        ], name=model_name + "_stem")

        # Downsampling layers
        self.downsample_layers = [self.stem]
        for i in range(3):
            self.downsample_layers.append(
                Sequential([
                    layers.Conv2D(projection_dims[i + 1], kernel_size=2, strides=2,
                                  name=model_name + f"_downsampling_conv_{i}")
                ], name=model_name + f"_downsampling_block_{i}")
            )

        # Drop rates for stochastic depth
        self.depth_drop_rates = np.linspace(0.0, drop_path_rate, sum(depths)).astype(float)

        # ConvNeXt stages
        self.stages = []
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlock(projection_dim=projection_dims[i],
                                  drop_path_rate=self.depth_drop_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_value,
                                  name_prefix=model_name + f"_stage_{i}_block_{j}")
                )
            self.stages.append(stage_blocks)
            cur += depths[i]

        # Latent projection if requested
        if latent_dim is not None:
            self.flatten = layers.Flatten()
            self.dense_proj = layers.Dense(256, activation="relu", name="dense_proj")
            self.z_mean = layers.Dense(latent_dim, name="z_mean")
            self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
            self.sampling = Sampling()

    def call(self, inputs, training=False):
        x = inputs
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x, training=training)

        if self.latent_dim is None:
            return x

        x = self.flatten(x)
        x = self.dense_proj(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return [z, z_mean, z_log_var]


class ConvNeXtDecoderTiny(Model):
    def __init__(self,
                 depths=[3, 9, 3, 3],
                 projection_dims=[768, 384, 192, 96],
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 model_name="convnext",
                 latent_dim=None):
        super().__init__(name=model_name)

        if latent_dim is None:
            raise ValueError("latent_dim must be specified for decoder")

        # Intro layer (dense + reshape)
        self.intro = Sequential([
            layers.Dense(10 * 10 * projection_dims[0], activation="relu"),
            layers.Reshape((10, 10, projection_dims[0]))
        ], name=model_name + "_intro")

        # Upsampling layers
        self.upsample_layers = [self.intro]
        for i in range(3):
            self.upsample_layers.append(
                Sequential([
                    layers.Conv2DTranspose(projection_dims[i + 1], kernel_size=2, strides=2,
                                           name=model_name + f"_upsampling_conv_{i}")
                ], name=model_name + f"_upsampling_block_{i}")
            )

        # Drop rates for stochastic depth
        self.depth_drop_rates = np.linspace(0.0, drop_path_rate, sum(depths)).astype(float)

        # ConvNeXt transpose stages
        self.stages = []
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlockTransposed(projection_dim=projection_dims[i],
                                            drop_path_rate=self.depth_drop_rates[cur + j],
                                            layer_scale_init_value=layer_scale_init_value,
                                            name_prefix=model_name + f"_stage_{i}_block_{j}")
                )
            self.stages.append(stage_blocks)
            cur += depths[i]

        # Top layer
        self.top = Sequential([
            layers.Conv2DTranspose(projection_dims[3], kernel_size=4, strides=4, name=model_name + "_top_conv")
        ], name=model_name + "_top")

        self.top_layer = TopLayer(filters=96)
        self.pred_layer = layers.Conv2DTranspose(3, kernel_size=1, activation="sigmoid",
                                                 padding="same", name="pred_layer")

    def call(self, inputs, training=False):
        x = inputs
        for i in range(4):
            x = self.upsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x, training=training)
        x = self.top(x)
        x = self.top_layer(x)
        return self.pred_layer(x)


if __name__ == "__main__":
    # Encoder
    encoder = ConvNeXtEncoderTiny(latent_dim=8)
    encoder.build((None, 320, 320, 3))
    encoder.summary()

    # Decoder
    decoder = ConvNeXtDecoderTiny(latent_dim=8)
    decoder.build((None, 8))
    decoder.summary()


