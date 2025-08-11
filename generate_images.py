import os

from argparse import ArgumentParser
from utils.inference_utils import init_vae_model, load_weights, get_noise_seeded, decode_noise, save_images


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, choices=['tide', 'tidev2'], help='VAE model')
    parser.add_argument("--weights_path", required=True, type=str, help='Path to restore trained weights')
    parser.add_argument("--latent_dim", default=8, type=int, help='Dimensionality of latent space')
    parser.add_argument("--save_dir", default="./fake_images", type=str, help='Path to save synthetic images')
    parser.add_argument("--num_of_images", default=10, type=int, help='Number of images to generate')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.weights_path):
        print("Not a valid path")

    vae = init_vae_model(args.model_name, args.latent_dim)
    noise_vector = get_noise_seeded((args.num_of_images, args.latent_dim))

    # Load weights
    vae = load_weights(vae, args.weights_path)
    vae.trainable = False

    # Generate & Save images
    fake_images = decode_noise(vae, noise_vector, return_list=True)
    save_images(args.save_dir, fake_images)