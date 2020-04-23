import os
import sys
import argparse
import numpy as np
from utils import stylegan_utils

def main():
    parser = argparse.ArgumentParser(description='Blend images with learned latent representations')
    parser.add_argument('first_latent', help='File with learned latent representation of the first character')
    parser.add_argument('second_latent', help='File with learned latent representation of the second character')
    parser.add_argument('result_img', help='Path to result directory')
    parser.add_argument('--blend_coeff', default=0.5,
                        help='The coefficient on mixing between the 2 characters, range between [0,1] default 0.5', type=float)
    parser.add_argument('--smile_coeff', default=0, help='The coefficient on smile latent direction, default 0', type=float)
    parser.add_argument('--gender_coeff', default=0, help='The coefficient on smile latent direction, default 0', type=float)
    parser.add_argument('--age_coeff', default=0, help='The coefficient on smile latent direction, default 0', type=float)
    args, other_args = parser.parse_known_args()
    
    # Load source latents
    if not os.path.exists(args.first_latent) or not os.path.exists(args.second_latent):
        return "Not all source latents exists"
    first_latent = np.load(args.first_latent)
    second_latent = np.load(args.second_latent)
    
    # Initalized the stylegan utils
    stylegan = stylegan_utils()
    generator, Gs  = stylegan.initUtils()

    # Loading already learned latent directions
    latentdirections_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffhq_dataset', 'latent_directions')
    smile_direction = np.load(os.path.join(latentdirections_dir, 'smile.npy'))
    gender_direction = np.load(os.path.join(latentdirections_dir, 'gender.npy'))
    age_direction = np.load(os.path.join(latentdirections_dir, 'age.npy'))

    char_latent = args.blend_coeff * first_latent.copy() + (1-(args.blend_coeff)) * second_latent.copy()
    char_latent += smile_direction * args.smile_coeff
    char_latent += gender_direction * args.gender_coeff
    char_latent += age_direction * args.age_coeff
    
    img = stylegan.generateImage(char_latent, 0, 0)
    img.save(args.result_img)
    print("Done: {}".format(args.result_img))


if __name__ == "__main__":
    ret = main()
    print(ret)
    sys.exit(0)
