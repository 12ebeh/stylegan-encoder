import os
import argparse
import numpy as np
from utils import stylegan_utils


def main():
    parser = argparse.ArgumentParser(description='Blend images with learned latent representations')
    parser.add_argument('first_latent', help='File with learned latent representation of the first character')
    parser.add_argument('second_latent', help='File with learned latent representation of the second character')
    parser.add_argument('result_video', help='Path to result video')
    parser.add_argument('--blend_direction', default='blend', help='One of blend, smile, age, gender. Default blend', type=str)
    parser.add_argument('--start_coeff', default=0.0,
                        help='The starting coefficient on video, range between [0,1] default 0', type=float)
    parser.add_argument('--end_coeff', default=1.0,
                        help='The ending coefficient on video, range between [0,1] default 1', type=float)
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
    smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
    gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
    age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
    
    # Blend Direction
    start_latent = first_latent.copy()
    stop_latent = second_latent.copy()
    if args.blend_direction == 'smile':
        start_latent += smile_direction * args.start_coeff
        stop_latent += smile_direction * args.end_coeff
    elif args.blend_direction == 'age':
        start_latent += age_direction * args.start_coeff
        stop_latent += age_direction * args.end_coeff
    elif args.blend_direction == 'gender':
        start_latent + gender_direction * args.start_coeff
        stop_latent += gender_direction * args.end_coeff

    # Video Generation
    stylegan.generateVideo(start_latent, stop_latent, outfile_name=args.result_video)
    return ("Done")


if __name__ == "__main__":
    ret = main()
    print(ret)
