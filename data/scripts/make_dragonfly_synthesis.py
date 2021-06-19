import os
import sys
import glob
import shutil
import cv2
import joblib
import random
import numpy as np
import skimage
import skimage.io
import skimage.transform
import skimage.filters
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def show(img):
    skimage.io.imshow(img)
    plt.show()




def imgutil_rotation(img):
    
    if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
    degreesCCW = random.uniform(-35, 35)
    scaleFactor = 1.0
    (oldY,oldX) = img.shape[0:2] #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty
    img = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    
    return img



def imgutil_overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]
    
    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]
    
    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
    
    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]
    
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / (1 + 255.0)
    
    background[y:y+h, x:x+w] = (1 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    return background


def imgutil_pileup_dragonfly(img_bg, img_fr):
    
    sq_size = img_fr.shape[0] if img_fr.shape[0] > img_fr.shape[1] else img_fr.shape[1]
    sq_size = sq_size + 10
    
    if (img_bg.shape[0] < sq_size) or (img_bg.shape[1] < sq_size):
        scl = random.uniform(1.5, 1.9)
        img_bg = cv2.resize(img_bg, (int(img_bg.shape[0] * scl), int(img_bg.shape[1] * scl)))
    
    x_from = random.randint(0, int(img_bg.shape[0] - sq_size))
    y_from = random.randint(0, int(img_bg.shape[1] - sq_size))
    
    img_bg = img_bg[x_from:(x_from + sq_size), y_from:(y_from + sq_size)]
    
    w_from = int((sq_size - img_fr.shape[1]) / 2)
    h_from = int((sq_size - img_fr.shape[0]) / 2)
    dest = imgutil_overlay_transparent(img_bg, img_fr, w_from, h_from)
    
    return dest
        
 

def imgutil_pileup_finger(img_bg, img_fr):
    w_from = random.randint(0, img_bg.shape[1] - img_fr.shape[1])
    h_from = random.randint(0, img_bg.shape[0] - img_fr.shape[0])
    dest = imgutil_overlay_transparent(img_bg, img_fr, w_from, h_from)
    
    return dest
        
       
    
 
def add_noise(img):
    r = np.random.rand(1)
    if r < 0.15:
        img = skimage.util.random_noise(img, mode='localvar')
    elif r < 0.30:
        img = skimage.util.random_noise(img, mode='salt')
    elif r < 0.45:
        img = skimage.util.random_noise(img, mode='s&p')
    elif r < 0.60:
        img = skimage.util.random_noise(img, mode='speckle', var=0.01)
    elif r < 0.75:
        img = skimage.util.random_noise(img, mode='poisson')
    elif r < 0.95:
        img = skimage.util.random_noise(img, mode='gaussian', var=0.01)
    img = img * 255
    img = img.astype(np.uint8)
    
    return img
   



def synthesis(input_path=None, bg_path=None, output_dirpath=None, n=100, output_prefix='synthetic_image', n_jobs=-1):
    
    mask_image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and (not f.startswith('.'))]
    
    _bg_path = os.path.join(bg_path, 'field')
    bg_field_files = [os.path.join(_bg_path, f) for f in os.listdir(_bg_path) if os.path.isfile(os.path.join(_bg_path, f)) and (not f.startswith('.'))]
    
    _bg_path = os.path.join(bg_path, 'finger')
    bg_finger_files = [os.path.join(_bg_path, f) for f in os.listdir(_bg_path) if os.path.isfile(os.path.join(_bg_path, f)) and (not f.startswith('.'))]
    
    
    def __synthesis_ss(i):
        
        try_next = True
        
        while try_next:
            try:
                # load images to objects
                mask_image_file = random.choice(mask_image_files)
                bg_field_file = random.choice(bg_field_files)
                bg_finger_file = random.choice(bg_finger_files)
        
                mask_image = skimage.io.imread(mask_image_file)
                bg_field_image = skimage.io.imread(bg_field_file)[:,:,:3]
                bg_finger_image = skimage.io.imread(bg_finger_file)
        
                img_dragonfly = imgutil_rotation(mask_image)
                img = imgutil_pileup_dragonfly(bg_field_image, img_dragonfly)
                if np.random.rand(1) > 0.5:
                    img = imgutil_pileup_finger(img, bg_finger_image)
        
                img = add_noise(img)
                
                img = skimage.transform.resize(img, (512, int(512 / img.shape[0] * img.shape[1])),
                                               anti_aliasing=False)
                img = img * 255
                img = img.astype(np.uint8)
                
                new_file_path = os.path.join(output_dirpath, output_prefix + '_' + str(i) + '.png')
                skimage.io.imsave(new_file_path, img)
                
                try_next = False
            except ValueError:
                print('faluire to synthesis image, try again.')
            except TypeError:
                print('faluire to synthesis image, try again.')
                
    
    r = joblib.Parallel(n_jobs=n_jobs, verbose=0)([joblib.delayed(__synthesis_ss)(i + 1) for i in range(n)])







def synthesis_main(mask_dirpath, bg_dirpath, output_dirpath, n_images):
    '''
    Synthesis images for training
    
    This function randomly samples a mask image of dragonfly and a background image,
    and synthesis both into one image.
    '''
    
    for d in sorted(glob.glob(os.path.join(mask_dirpath, '*'))):
        print(d)
        
        synimage_dirpath = os.path.join(output_dirpath, os.path.basename(d))
        
        if not os.path.exists(synimage_dirpath):
            os.makedirs(synimage_dirpath)
        
        synthesis(d, bg_dirpath, synimage_dirpath, n=n_images, n_jobs=-1)
        
   


if __name__ == '__main__':
    
    mask_dpath = sys.argv[1]
    bg_dpath = sys.argv[2]
    output_dpath = sys.argv[3]
    n_images = int(sys.argv[4])
    
    synthesis_main(mask_dpath, bg_dpath, output_dpath, n_images)
    




      


