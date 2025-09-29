import os
import numpy as np
import tensorflow as tf

from PIL import Image

from model.vae import VAE
from model import tidev2


def init_vae_model(model_name, latent_dim, input_shape):
    if model_name == 'tidev2':
        vae_model = VAE(tidev2.ConvNeXtEncoderTiny(latent_dim=latent_dim),
                        tidev2.ConvNeXtDecoderTiny(latent_dim=latent_dim, image_dims=input_shape[:2], out_channels=input_shape[-1])
                       )
        vae_model.build((None, *input_shape))
        return vae_model


def load_weights(vae, weights_path):
    print("Loading weights from {}".format(weights_path))
    if "ckpt-" in weights_path:
        ckpt = tf.train.Checkpoint(vae=vae)
        ckpt.restore(weights_path).expect_partial()
        return vae
    if ".TF" in weights_path:
        vae.load_weights(weights_path, by_name=True)
        return vae


def get_noise_seeded(noise_shape, seed=0):
    np.random.seed(seed)
    random_z = np.random.normal(0, 1, noise_shape)
    return random_z

def decode_noise(trained_vae, noise, return_list=False):
    print("Generating synthetic images ...")
    pred = trained_vae.decoder.predict(noise, batch_size=1)
    # print(type(pred), pred.shape, pred.dtype, pred.min(), pred.max())
    pred *= 255.0
    # print(type(pred), pred.shape, pred.dtype, pred.min(), pred.max())
    if return_list:
        return [img for img in pred]
    return pred


def save_images(save_folder, images, seed=None):
    print(f"Saving  synthetic images into {save_folder}")
    if isinstance(images, list):
        for i, image in enumerate(images):
            image = image.astype(np.uint8)
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            save_filename = f"image-{i}.jpg" if seed is None else f"image-{seed}.jpg"
            Image.fromarray(image).save(os.path.join(save_folder, save_filename))