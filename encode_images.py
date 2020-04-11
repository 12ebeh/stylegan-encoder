import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def run(src_dir, generated_images_dir, dlatent_dir,
    batch_size=1, image_size=256, lr=1, iterations=1000, randomize_noise=False):
    """
    Find latent representation of reference images using perceptual loss
    Params:
        src_dir: Directory for storing genrated images
        generated_images_dir: Directory for storing generated images
        dlatent_dir: Directory for storing dlatent representations
        batch_size: Batch size for generator and perceptual model
        image_size: Size of images for perceptual model
        lr: Size of images for perceptual model
        iterations: Number of optimization steps for each batch
        randomize_noise: Add noise to dlatents during optimization
    """
    ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % src_dir)

    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
        pbar = tqdm(op, leave=False, total=iterations)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print(' '.join(names), ' loss:', loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()
