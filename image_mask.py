import numpy as np
import cv2
from glob import glob
from PIL import Image 

def masks(type, tt):

    images = np.sort(glob('Figaro1k/Original/'+tt+'/'+type+'/*'))
    masks = np.sort(glob('Figaro1k/GT/'+tt+'/'+type+'/*'))

    for i, img in enumerate(images):

        fn = img.split('/')[-1]

        im = cv2.imread(img)
        mk = cv2.imread(masks[i])

        new_img = im * mk

        cv2.imwrite('Figaro1k/Combo/'+tt+'/'+type+'/'+fn, new_img)


    return
