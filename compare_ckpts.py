import sys
import glob
import numpy as np
import pandas as pd
import importlib
import tensorflow as tf

from PIL import Image
from re import split, compile
from tensorflow.keras.applications.inception_v3 import preprocess_input

import fid_kid

#TODO : uncomment & import appropriate paths if msgastrovae_smc.py (TIDE model) & my_convnext.py (TIDE-2 model)
#       are not in the same folder with this script (for importlib modules below)
# sys.path.append(f'{tide_path}')
# sys.path.append(f'{tide2_path}')


def list_saved_models(results_dir):
    models_found = glob.glob("{}/weights/vae_checkpoints/*.index".format(results_dir))    # checkpoints
    models_found.extend(glob.glob("{}/weights/*.h5".format(results_dir)))                 # models
    models_found.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in split(compile(r'(\d+)'), l)])
    # print(models_found)
    return models_found


def get_real_kid_filenames(label):
    kid_dir = '/mnt/storage/shared/ckaitanidis/datasets/kid/kid-dataset-2/'
    files = []
    if label in ['inflammatory', 'polypoid', 'vascular']:
        files = glob.glob('{}/{}/*.png'.format(kid_dir, label))
    elif label == 'normaleso':
        files = glob.glob('{}/normal-esophagus'.format(kid_dir))
    elif label == 'normalstom':
        files = glob.glob('{}/normal-stomach'.format(kid_dir))
    elif label == 'normalcolon':
        files = glob.glob('{}/normal-colon'.format(kid_dir))
    elif label == 'normalsb':
        files = glob.glob('{}/normal-small-bowel'.format(kid_dir))
    print('Real images found: {}'.format(len(files)))
    files = sorted(files)
    return files


def init_vae_model(model_name, latent_dim, input_shape):
    if model_name == 'tide':
        vae = importlib.import_module("msgastrovae_smc")
        vae_model = vae.VAE(vae.create_encoder(latent_dim=latent_dim,input_shape=input_shape),
                            vae.create_decoder(latent_dim=latent_dim))
        vae_model.build(input_shape=[(None,) + input_shape])
        return vae_model
    elif model_name == 'tide2':
        vae = importlib.import_module("my_convnext")
        vae_model = vae.VAE(vae.create_encoder_tiny(latent_dim=latent_dim, input_shape=input_shape),
                            vae.create_decoder_tiny(latent_dim=latent_dim))
        vae_model.build(input_shape=[(None,) + input_shape])
        return vae_model


def load_weights(vae, weights_path):
    print("Loading weights from {}".format(weights_path))
    if "ckpt-" in weights_path:
        weights_path = weights_path.split(".index")[0]
        ckpt = tf.train.Checkpoint(vae=vae)
        status = ckpt.restore(weights_path).expect_partial()
        return vae
    if ".h5" in weights_path:
        vae.load_weights(weights_path, by_name=True)
        return vae


def debug_weights_loading(vae):
    decoder_weights = vae.decoder.get_weights()
    print("Decoder layer 0 weights shape:", decoder_weights[0].shape)
    print("Decoder layer 0 weights sample:", decoder_weights[0].flatten()[:5])
    # 'Decoder layer 0 weights sample: [-0.01202846 -0.02691004  0.00642165 -0.02967337 -0.03743371]' # mine always same


def get_noise_seeded(noise_shape):
    np.random.seed(0)
    random_z = np.random.normal(0, 1, noise_shape)
    return random_z


def decode_noise(trained_vae, noise, return_list=False):
    print("Generating fake images ...")
    pred = trained_vae.decoder.predict(noise, batch_size=1)
    # print(type(pred), pred.shape, pred.dtype, pred.min(), pred.max())
    pred *= 255.0   # for tf.preprocess_input requires [0, 255]
    # print(type(pred), pred.shape, pred.dtype, pred.min(), pred.max())
    if return_list:
        return [img for img in pred]
    return pred


def visualize_debug(image, name='output.png'):
    image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(name)


def kid_dataset_center_crop(img, crop_size=(320, 320)):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    h, w, _ = img.shape
    ch, cw = crop_size

    top = (h - ch) // 2
    left = (w - cw) // 2

    return img[top:top + ch, left:left + cw]


if __name__ == "__main__":

    # Change these two to run across all models
    model_name = 'tide2'             # 'tide'  'tide2'
    label = "inflammatory"           # 'inflammatory' 'vascular' 'polypoid' 'normaleso' 'normalcolon' 'normalsb' 'normalstom'

    results_dir = "/mnt/storage/shared/ckaitanidis/"
    if model_name == 'tide':
        results_dir += 'kid_latent6_sr96_50000ep_{}'.format(label)
    elif model_name == 'tide2':
        results_dir += 'kid_latent8_tide2_sr96_50000ep_{}'.format(label)
    print(results_dir)

    # Params auto
    latent_dim = 6 if model_name == 'tide' else 8
    input_shape = (96, 96, 3) if model_name == 'tide' else (256, 256, 3)
    crop_dim = (320, 320)

    trained_weights = list_saved_models(results_dir)
    real_filenames = get_real_kid_filenames(label)
    real_images = fid_kid.get_images_inception(real_filenames, crop_dim=crop_dim)  # this returns np.array, float32, [-1, 1], (batch, 299, 299, 3)
    # print(type(real_images), real_images.shape, real_images.dtype, real_images.min(), real_images.max())
    visualize_debug(real_images[0], name='output1.png')

    vae = init_vae_model(model_name, latent_dim, input_shape)
    noise_vector = get_noise_seeded((len(real_filenames), latent_dim))


    results = []

    # Ignore these - my weights for debug
    # trained_weights = ['/mnt/storage/pgatoula-private/codes/tide-panagiota/results_kid/kid_inflammatory_latent6/weights/vae_checkpoints/ckpt-4500.index']
    # trained_weights = ['/mnt/storage/pgatoula-private/general-results/convnext/kid_inflammatory_latent8_lbfcn_sr96/weights/vae_checkpoints/ckpt-1400.index',
    #                    '/mnt/storage/pgatoula-private/general-results/convnext/kid_inflammatory_latent8_lbfcn_sr96/weights/vae_checkpoints/ckpt-1600.index']

    for weights in trained_weights:
        # Load weights
        vae = load_weights(vae, weights)
        vae.trainable = False
        # try:
        #     debug_weights_loading(vae)
        # except Exception as e:
        #     print(f"Skipping {weights} due to load failure: {e}")
        #     continue

        # Generate Fakes
        fake_images = decode_noise(vae, noise_vector, return_list=True)
        fake_images = preprocess_input(fake_images)
        # print(type(fake_images), fake_images.shape, fake_images.dtype, fake_images.min(), fake_images.max())
        fake_images = tf.image.resize(fake_images, size=(299, 299), method='bicubic').numpy()
        # print(type(fake_images), fake_images.shape, fake_images.dtype, fake_images.min(), fake_images.max())
        visualize_debug(fake_images[0], name='output2.png')

        # Calculate metrics
        fid_score = fid_kid.calculate_fid(real_images, fake_images)
        kid_mean, kid_std = fid_kid.calculate_kid(real_images, fake_images)

        fid_score = round(fid_score, 4)
        kid_mean = round(kid_mean, 4)
        kid_std = round(kid_std, 4)

        print("{}: FID={} KID={} ± {}".format(weights, fid_score, kid_mean, kid_std))

        results.append({'weights': weights,
                        'fid': fid_score,
                        'kid_mean': kid_mean,
                        'kid_std': kid_std,
                        })

    # Save in xlxs
    df = pd.DataFrame(results)
    excel_path = f"ckpt_metrics_{model_name}.xlsx"

    # TODO: pip install XlsxWriter if not installed
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=label, index=False)

    print(f"Results saved to: {excel_path}")