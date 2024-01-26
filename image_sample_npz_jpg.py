"""
    File contains scripts for converting .npz file archive created from
    'image_sample.py' to folder full of .JPEG iamge samples used for 
    computing fid with 'fid_score.py'.
"""

import numpy as np
from PIL import Image
import os

def convert_npz_to_jpegs(npz_file, output_folder):
    # Load the .npz file
    data = np.load(npz_file)
    # Extract images from the 'images' key
    images = data['arr_0']

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save each image as a JPEG in the output folder
    for i, image in enumerate(images):
        # Assuming images are in uint8 format
        image_path = os.path.join(output_folder, f'image_{i + 1}.jpeg')
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)
        # Save the PIL Image as a JPEG
        pil_image.save(image_path)

    print(f"Conversion completed. {len(images)} images saved to {output_folder}.")


def main():
    # Specify the path to your .npz file and the output folder
    npz_file_path = '/home/datasets/fid_samples_2/samples_50000x28x28x3.npz'
    output_folder_path = '/home/datasets/fid_samples_2/'

    # Call the function to perform the conversion
    convert_npz_to_jpegs(npz_file_path, output_folder_path)


if __name__=="__main__":
    main()
