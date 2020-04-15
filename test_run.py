import numpy as np
from utils import stylegan_utils

stylegan = stylegan_utils()
generator, Gs  = stylegan.initUtils()

# Loading already learned representations
jon = np.load('latent_representations/jon.npy')
jon_2 = np.load('latent_representations/jon_2.npy')
tyrion = np.load('latent_representations/tyrion.npy')
daenerys = np.load('latent_representations/daenerys.npy')
drogo = np.load('latent_representations/drogo.npy')
bran = np.load('latent_representations/bran.npy')
cersei = np.load('latent_representations/cersei.npy')
jaime = np.load('latent_representations/jaime.npy')
theon = np.load('latent_representations/theon.npy')
night_king_1 = np.load('latent_representations/night_king_1.npy')
night_king_2 = np.load('latent_representations/night_king_2.npy')

print(jon.shape)
print(len(jon.reshape((1,18,512))))

# Loading already learned latent directions
smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
age_direction = np.load('ffhq_dataset/latent_directions/age.npy')

# Single Image Generation
smile = [
    stylegan.generateImage(jaime.copy(), smile_direction, -1.0),
    stylegan.generateImage(jaime.copy(), smile_direction, 0.0),
    stylegan.generateImage(jaime.copy(), smile_direction, 1.0)
]

i = -1
for img in smile:
    img.save('smile_{}.png'.format(i))
    i += 1

# Multi Image Generation
stylegan.generateImages(daenerys.copy(), age_direction, [-2, 0, 2], 'smile2')

# Video Generation
stylegan.generateVideo(daenerys.copy(), jon.copy(), outfile_name='results/video')

# Draw Style Mixing Image
#_Gs_cache = dict()
#stylegan.drawStyleMixingImage(1024, 1024,
#                              src_dlatents=daenerys.reshape((1, 18, 512)),
#                              dst_dlatents=jon_2.reshape((1, 18, 512)), 
#                              style_ranges=[range(6,14)], outfile_name='style_mix')

# Align Images
#stylegan.alignImages('raw_images', 'aligned_images')

# Encode Images
#stylegan.encodeImages('aligned_images', 'generated_images', 'latent_representations')