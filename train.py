import os
import tensorflow as tf

from json import dump
from argparse import ArgumentParser


from model import tidev2
from model.vae import VAE
from utils.callbacks import VisualizeCallback, CheckpointCallback
from utils.dataloader import list_filenames, Dataset
from utils.plots import visualize_from_latent_space


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, choices=['tide', 'tidev2'], help='VAE model')
    parser.add_argument("--output_path", default='./results/', type=str, help='Path to store the results')
    # VAE model
    parser.add_argument("--input_shape", default=(320, 320, 3), type=tuple, help='Image shape for training')
    parser.add_argument("--dim_latent", default=8, type=int, help='Dimensionality of latent space')
    # Training
    parser.add_argument("--epochs", default=5000, type=int, help='Number of training epochs')
    parser.add_argument("--batch_size", default=4, type=int, help='Number of training batch size')
    parser.add_argument("--learning_rate",  default=0.0002, type=float, help='Learning rate')
    parser.add_argument("--ckpt_interval", default=200, type=int, help='Epoch interval for saving checkpoints')
    parser.add_argument("--visualization_interval", default=25, type=int, help='Epoch interval for visualizing results')
    # Data
    parser.add_argument("--datadir", default='./kid/inflammatory', type=str, help='Folder with images for training')
    parser.add_argument("--files_ext", default='png', type=str, help='Extension of training files')
    parser.add_argument("--files_prefix", default=None, type=str,
                        help='Prefix of training files. Ignore if datadir contains only the appropriate files')
    parser.add_argument("--crop_dim", default=None, type=tuple,
                        help='Dimensions for cropping images. Ignore if images are already cropped')
    args = parser.parse_args()

    # Create folders & Save training config
    os.makedirs(args.output_path, exist_ok=True)
    log_dir = os.path.join(args.output_path, 'logs')
    ckpt_dir = os.path.join(args.output_path, 'checkpoints')
    visualize_dir = os.path.join(args.output_path, 'visualize')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)

    with open(os.path.join(args.output_path, "training_config.json"), 'w') as fp:
        dump(vars(args), fp)

    # Setup training data
    filenames = list_filenames(data_path=args.datadir,
                               img_extension=args.files_ext,
                               filename_prefix=args.files_prefix)
    images = Dataset(filenames,
                     batch_size=args.batch_size,
                     crop_dim=args.crop_dim,
                     resize_dim=args.input_shape[:2],)

    # Create Model
    if args.model_name == 'tidev2':
        vae = VAE(tidev2.ConvNeXtEncoderTiny(latent_dim=args.dim_latent),
                  tidev2.ConvNeXtDecoderTiny(latent_dim=args.dim_latent)
                  )
        vae.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    # Training
    callbacks = [VisualizeCallback(args.visualization_interval, lambda model,  epoch: visualize_from_latent_space(
                                                                                latent_dim=args.dim_latent,
                                                                                input_shape=args.input_shape,
                                                                                vae=model,
                                                                                output_path=visualize_dir,
                                                                                epoch=epoch,
                                                                                num_items=10,)),
                 CheckpointCallback(vae=vae,
                                    path=ckpt_dir,
                                    epoch_interval=args.ckpt_interval,
                                    restore_training=False,
                                    restore_path=None),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    vae.fit(x=images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            shuffle=True,
            initial_epoch=0)

