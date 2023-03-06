from pathlib import Path
from PIL import Image
import numpy as np

from utils import DataExtractor
from utils import MaskGenerator


def draw_grey_skirt(file_num, mask):
    """Drawing grey pixs from given mask on image with given file number"""
    path = f'./dataset/image/{file_num}_00.jpg'
    im = Image.open(path)
    im_pix = np.array(im)
    # Applying grey mask
    im_pix[mask] = [128, 128, 128]
    im_new = Image.fromarray(im_pix)
    return im_new


if __name__ == "__main__":
    # Create object for extracting data from dataset (masks, json)
    data_extr = DataExtractor()
    # Create object of class that has all methods for mask generation
    mask_generator = MaskGenerator(data_extr)
    
    path_dir = Path.cwd() / 'imgs'
    path_dir.mkdir(exist_ok=True)
   
    path = Path('./dataset/image')    
    for file in path.iterdir():
        # Extracting file number
        file_num = file.stem[:-3]
        
        # Creating mask for big skirt
        mask = mask_generator.calc_result_mask(file_num)
        im = draw_grey_skirt(file_num, mask)
        
        # Saving result img
        im.save(f'./imgs/{file_num}_00.png')