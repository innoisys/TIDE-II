import numpy as np
from PIL import Image


def visualize_from_latent_space(latent_dim, input_shape, vae, output_path, epoch="final", num_items=10,):

    image_size, _, img_channels = input_shape
    figure = np.zeros((image_size * num_items, image_size * num_items, img_channels))

    scale = 1.0
    grid_x = np.linspace(-scale, scale, num_items)
    grid_y = np.linspace(-scale, scale, num_items)[::-1]

    np.random.seed(42)
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            random_z = np.random.normal(0, 1, (1, latent_dim))
            x_decoded = vae.decoder.predict(random_z)
            image = x_decoded[0].reshape(input_shape)
            figure[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size, ] = image
    print(f'Saving collage in {output_path}/decoding-noise-ep{epoch}.png')
    figure = (figure * 255).astype('uint8')
    if img_channels == 1:
        figure = np.squeeze(figure, axis=-1)
    figure = Image.fromarray(figure)
    figure.save(f"{output_path}/decoding-noise-ep{epoch}.jpg")

