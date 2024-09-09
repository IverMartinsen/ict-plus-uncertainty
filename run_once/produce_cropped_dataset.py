import os
import numpy as np
from PIL import Image

source_path = './data/Man vs machine_Iver/S/'
dest_path = './data/Man vs machine_Iver_cropped/S/'


def extract_center_crop(image, border_size=5):
    # find the rectangle at the center of the image that encompasses a foreground object
    midpoint = image.shape[0] // 2, image.shape[1] // 2
    horizontal_lims = np.where(image[midpoint[1], border_size:-border_size, :].sum(axis=1) != 0)[0] + border_size
    vertical_lims = np.where(image[border_size:-border_size, midpoint[0], :].sum(axis=1) != 0)[0] + border_size
    left, right, top, bottom = horizontal_lims[0], horizontal_lims[-1], vertical_lims[0], vertical_lims[-1]
    return image[top:bottom, left:right, :]


if __name__ == '__main__':
    filenames = os.listdir(source_path)

    os.makedirs(dest_path, exist_ok=True)

    for filename in filenames:
        img = Image.open(source_path + filename)
        img = np.array(img)
        img = extract_center_crop(img)
        img = Image.fromarray(img)
        img.save(dest_path + filename)
        print('Saved', dest_path + filename)
