import numpy as np
import cv2
from glob import glob
from PIL import Image 

def masks():

    images = glob('Figaro1k/Original/Training/straight/*')
    masks = glob('Figaro1k/GT/Training/straight/*')

    for i, img in enumerate(images):

        fn = img.split('/')[-1]

        im = cv2.imread(img)
        mk = cv2.imread(masks[i])

        new_img = im * mk

        cv2.imwrite('Figaro1k/Combo/Training/straight/'+fn, new_img)


    return
