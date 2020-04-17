import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import moviepy.editor
import pickle
import PIL.Image
import bz2
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from tqdm import tqdm
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

class stylegan_utils:
    URL_FFHQ = 'https://s3-us-west-2.amazonaws.com/nanonets/blogs/karras2019stylegan-ffhq-1024x1024.pkl'
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    
    
    # Initialization Functions
    def initUtils(self):
        tflib.init_tf()
        with dnnlib.util.open_url(stylegan_utils.URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs = pickle.load(f)

        self.generator = Generator(Gs, batch_size=1, randomize_noise=False)
        self.Gs = Gs
        return self.generator, self.Gs

    
    # Image Generation Functions
    def generateImage(self, latent_vector, direction, coefficient):
        vec = latent_vector.copy() + direction * coefficient
        return self._generateImage(latent_vector)


    def _generateImage(self, latent_vector):
        latent_vector = latent_vector.reshape((1, 18, 512))
        self.generator.set_dlatents(latent_vector)
        img_array = self.generator.generate_images()[0]
        img = PIL.Image.fromarray(img_array, 'RGB')
        return img.resize((256, 256))


    def generateImages(self, latent_vector, direction, coefficients, outfile_name=None, outfile_format='png'):
        l = []
        print(coefficients)
        for coeff in coefficients:
            img = self.generateImage(latent_vector, direction, coeff)
            if outfile_name is not None:
                img.save("{out}_{c}.{fmt}".format(out=outfile_name, c=coeff, fmt=outfile_format))
            l.append(img)
        return l
    
    
    # Video Functions
    def generateVideo(self, first_latent, second_latent,
                      duration_sec=5.0, smoothing_sec=1.0, fps=24, codec='libx264', bitrate='5M',
                      outfile_name=None, outfile_format='mp4'):
        num_frames = int(np.rint(duration_sec * fps))
        
        vid_file = None
        if outfile_name is not None:
            vid_file = '{out}.{fmt}'.format(out=outfile_name, fmt=outfile_format)
        
        src_images = np.stack(self._generateImage((0.01*alpha*second_latent)+((1-(0.01*alpha))*first_latent)) for alpha in range (num_frames))
        
        def make_frame(t):
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            src_image = src_images[frame_idx]
            return np.array(src_image)
        
        video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
        if vid_file is not None:
            video_clip.write_videofile(vid_file, fps=fps, codec=codec, bitrate=bitrate)
        return video_clip


    # Style Mixing Functions - Not working atm
    def drawStyleMixingImage(self, w, h, src_dlatents, dst_dlatents, style_ranges, outfile_name=None, outfile_format='png', randomize_noise=False):
        tflib.init_tf()
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)
        
        png = None
        if outfile_name is not None:
            png = "{out}.{fmt}".format(out=outfile_name, fmt=outfile_format)
        print(png)

        src_images = self.Gs.components.synthesis.run(src_dlatents, randomize_noise, **synthesis_kwargs)
        dst_images = self.Gs.components.synthesis.run(dst_dlatents, randomize_noise, **synthesis_kwargs)

        canvas = PIL.Image.new('RGB', (w * (len(src_dlatents) + 1), h * (len(dst_dlatents) + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
            row_dlatents = np.stack([dst_dlatents[row]] * len(src_dlatents))
            row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
            row_images = self.Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
            for col, image in enumerate(list(row_images)):
                canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(png)
        return canvas.resize((512,512))
    
    
    # Align Image Functions
    def _unpack_bz2(self, src_path):
        data = bz2.BZ2File(src_path).read()
        dst_path = src_path[:-4]
        with open(dst_path, 'wb') as fp:
            fp.write(data)
        return dst_path
    
    def alignImages(self, raw_img_dir, aligned_img_dir):
        RAW_IMAGES_DIR = raw_img_dir
        ALIGNED_IMAGES_DIR = aligned_img_dir
        
        self.landmarks_model_path = self._unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                         stylegan_utils.LANDMARKS_MODEL_URL, cache_subdir='temp'))

        landmarks_detector = LandmarksDetector(self.landmarks_model_path)
        for img_name in os.listdir(RAW_IMAGES_DIR):
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

                image_align(raw_img_path, aligned_face_path, face_landmarks)


    # Encode Image Functions
    def _split_to_batches(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def encodeImages(self, src_dir, generated_images_dir, dlatent_dir,
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

        perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
        perceptual_model.build_perceptual_model(self.generator.generated_image)

        # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
        for images_batch in tqdm(self._split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

            perceptual_model.set_reference_images(images_batch)
            op = perceptual_model.optimize(self.generator.dlatent_variable, iterations=iterations, learning_rate=lr)
            pbar = tqdm(op, leave=False, total=iterations)
            for loss in pbar:
                pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
            print(' '.join(names), ' loss:', loss)

            # Generate images from found dlatents and save them
            generated_images = self.generator.generate_images()
            generated_dlatents = self.generator.get_dlatents()
            for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
                img = PIL.Image.fromarray(img_array, 'RGB')
                img.save(os.path.join(generated_images_dir, f'{img_name}.png'), 'PNG')
                np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), dlatent)

            self.generator.reset_dlatents()


def generate_image_for_video(generator, latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]

    return img_array


def move_for_video(generator, latent_vector, direction, coeff):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    img_array = generate_image(new_latent_vector)
    return img_array
