# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import argparse
import glob
from PIL import Image
from tqdm import tqdm


def run(path, height, width):
    """Resize images"""
    print('resizing images')
    all_images = glob.glob(path + '*')
    for image_file in tqdm(all_images):
        im = Image.open(image_file)
        im = im.resize((height, width), resample=Image.LANCZOS)
        im.save(image_file)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to resize a set of images in a folder')
    parser.add_argument("-f", "--folder", type=str, help="Path to folder containing the images to resize " +
                        "(it can contain the special character '*' to refer to a set of folders).")
    parser.add_argument("-H", "--height", type=int, help="Height of the resized image")
    parser.add_argument("-W", "--width", type=int, help="Width of the resized image")
    args = parser.parse_args()
    run(args.folder, args.height, args.width)
