import os
import argparse
import numpy as np
from server.style_encoder.utils import stylegan_utils


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_img_dir', help='Directory with images for encoding')
    parser.add_argument('aligned_img_dir', help='Directory with aligned images')
    parser.add_argument('generated_img_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
    args, other_args = parser.parse_known_args()
    
    # Check directories
    if not os.path.isdir(args.src_img_dir):
        return "src_img_dir is not a directory"
    
    if not os.path.isdir(args.aligned_img_dir):
        os.mkdir(args.aligned_img_dir)
    
    if not os.path.isdir(args.generated_img_dir):
        os.mkdir(args.generated_img_dir)
    
    if not os.path.isdir(args.dlatent_dir):
        os.mkdir(args.dlatent_dir)
    
    # Initalized the stylegan utils
    stylegan = stylegan_utils()
    generator, Gs  = stylegan.initUtils()
    
    # Align Images
    stylegan.alignImages(args.src_img_dir, args.aligned_img_dir)

    # Encode Images
    stylegan.encodeImages(args.aligned_img_dir, args.generated_img_dir, args.dlatent_dir)
    
    return "All images trained, latents at {}".format(args.dlatent_dir)

if __name__ == "__main__":
    ret = main()
    print(ret)