"""
For each class, load images and save as numpy arrays.
"""

import numpy as np
import os
import pickle
import scipy.misc
import pandas as pd
import scipy
from tqdm import tqdm
from PIL import Image

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)

def custom_crop(img: np.array, bbox: np.array):
    # bbox = [x-left, y-top, width, height]
    imsiz = img.shape  # [height, width, channel]
    # if box[0] + box[2] >= imsiz[1] or\
    #     box[1] + box[3] >= imsiz[0] or\
    #     box[0] <= 0 or\
    #     box[1] <= 0:
    #     box[0] = np.maximum(0, box[0])
    #     box[1] = np.maximum(0, box[1])
    #     box[2] = np.minimum(imsiz[1] - box[0] - 1, box[2])
    #     box[3] = np.minimum(imsiz[0] - box[1] - 1, box[3])
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    return img_cropped

def transform(image: Image, image_size: int, bbox: np.array):
    image = custom_crop(np.array(image), bbox)
    transformed_image = Image.fromarray(image).resize([image_size, image_size], resample=Image.BICUBIC)
    return np.array(transformed_image)

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Save numpy", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cub_dir", default="CUB_200_2011/images", help="Directory to load/cache"
    )
    parser.add_argument(
        "--original_cub_dir",
        default="CUB_200_2011/images",
        help="Original CUB directory if you want the image keys to be different (in case --cub_dir has changed)",
    )
    parser.add_argument("--filelist_prefix", default="./filelists/CUB/")
    parser.add_argument("--skip_preprocess", action='store_true')

    args = parser.parse_args()

    if args.skip_preprocess:
        # load bounding boxes into filename_bbox
        bbox_path = os.path.join(args.filelist_prefix, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(args.filelist_prefix, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])

        filename_bbox = {img_file: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i]
            filename_bbox[key] = bbox

        # filename_bbox dict: filename -> bounding box coordinates

    for bird_class in tqdm(os.listdir(args.cub_dir), desc="Classes"):
        bird_imgs_np = {}
        class_dir = os.path.join(args.cub_dir, bird_class)
        bird_imgs = sorted([x for x in os.listdir(class_dir) if x != "img.npz"])
        for bird_img in bird_imgs:
            bird_img_fname = os.path.join(class_dir, bird_img)
            img = Image.open(bird_img_fname).convert("RGB")

            if not args.skip_preprocess:
                transform(img, LOAD_SIZE, filename_bbox[bird_img_fname])

            full_bird_img_fname = os.path.join(
                args.filelist_prefix, args.original_cub_dir, bird_class, bird_img
            )

            img_np = np.asarray(img)
            bird_imgs_np[full_bird_img_fname] = img_np

        np_fname = os.path.join(class_dir, "img.npz")
        np.savez_compressed(np_fname, **bird_imgs_np)
