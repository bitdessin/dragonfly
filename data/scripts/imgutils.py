import os
import sys
import re
import glob
import errno
import shutil
import random
import numpy as np
import cv2
import skimage
import skimage.io
import skimage.transform
import skimage.filters
import joblib



class imgUtils:
    
    
    
    def __init__(self):
        
        self.image_extension = ['.jpeg', '.jpg', '.png', '.tif', '.tiff',
                                '.JPEG', '.JPG', '.PNG', '.TIF', '.TIFF']
    
    
    
    def __augmentation_rotation(self, img):
        r = np.random.rand(1)
        random_degree = random.uniform(0, 90)
        img = skimage.transform.rotate(img, random_degree, resize=True, cval=0)
        return img
    
    
    def __augmentation_flip(self, img):
        r = np.random.rand(1)
        if r < 1/3:
            img = img[:, ::-1, :]
        elif r < 2/3:
            img = img[::-1, :, :]
        return img
    
    
    def __augmentation_noise(self, img):
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
    
    
    def __augmentation_generate_background(self, img):
        bg_img = self.__augmentation_flip(img)
        
        # crop background image
        x0 = random.randint(0, int(bg_img.shape[0] / 3))
        x1 = random.randint(int(2 * bg_img.shape[0] / 3), bg_img.shape[0])
        y0 = random.randint(0, int(bg_img.shape[1] / 3))
        y1 = random.randint(int(2 * bg_img.shape[1] / 3), bg_img.shape[1])
        bg_img = bg_img[x0:x1, y0:y1]
            
        # resize background image
        w = random.randint(int(bg_img.shape[0] * 5), bg_img.shape[0] * 8)
        h = random.randint(int(bg_img.shape[1] * 5), bg_img.shape[1] * 8)
        bg_img = skimage.transform.resize(bg_img, (w, h), mode='constant')
        
        # rotate background image
        bg_img = self.__augmentation_rotation(bg_img)
       
        # filter
        r = np.random.rand(1)
        if r > 0.5:
            bg_img = skimage.filters.gaussian(bg_img, sigma=random.uniform(0.5, 2.0), multichannel=True)
        
        return bg_img
 


       
    def __zero_padding(self, v, w, iaxis, kwargs):
        v[:w[0]] = 0.0
        v[-w[1]:] = 0.0
        return v
            
        
    def __get_padding(self, img):
        h, w, c = img.shape
        longest_edge = max(h, w)
        top = 0
        bottom = 0
        left = 0
        right = 0
        if h < longest_edge:
            diff_h = longest_edge - h
            top = diff_h // 2
            bottom = diff_h - top
        elif w < longest_edge:
            diff_w = longest_edge - w
            left = diff_w // 2
            right = diff_w - left
        else:
            pass
         
        img_constant = cv2.copyMakeBorder(img, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])   
        return img_constant
        
        
        
    def __augmentation_fill_background(self, img, bg_img_tmpl):
        
       
        
        block_size = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]
        
        # get background and crop it to nxn sizes block
        bg_img = self.__augmentation_generate_background(bg_img_tmpl)
        x0 = int((bg_img.shape[0] - block_size) / 2)
        x1 = x0 + block_size
        y0 = int((bg_img.shape[1] - block_size) / 2)
        y1 = y0 + block_size
        bg_img = bg_img[x0:x1, y0:y1]
        
        img = self.__get_padding(img)
        
        # make mask
        mask = skimage.color.rgb2gray(img)
        mask = np.pad(skimage.transform.resize(mask, (mask.shape[0] - 10, mask.shape[1] - 10), mode='constant'),
                      5, self.__zero_padding)
        
        img[mask < 0.001] = bg_img[mask < 0.001]
        return img
        
        
    
    
    def augmentation_ss(self, input_path=None, output_dirpath=None, n=100, output_prefix='augmented_image'):
        
        
        if os.path.isfile(input_path):
            image_files = [input_path]
        elif os.path.isdir(input_path):
            image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        else:
            raise ValueError('Unknown types of this file: ' + input_path + '.')
        
        n = n + 1
        i = 1
        while i < n:
            
            # randomly chose an image from the directory
            image_path = random.choice(image_files)
            
            img = skimage.io.imread(image_path)[:, :, :3]
            
            # randomly rotation
            img_ag = self.__augmentation_rotation(img)
            
            # fill up background (some case can not perform zero-padding)
            _img_ag = self.__augmentation_fill_background(img_ag, img)
            if _img_ag is None:
                print('==> aug found None image in __augmentation_fill_background: ' + input_path + '.')
                _img_ag = img_ag
            img_ag = _img_ag

           
            # randomly reflection
            img_ag = self.__augmentation_flip(img_ag)
            
            # randomly add noises
            img_ag = self.__augmentation_noise(img_ag)
            
            new_file_path = os.path.join(output_dirpath, output_prefix + '_' + str(i) + '.png')
            skimage.io.imsave(new_file_path, img_ag)
            
            i = i + 1



    
    def augmentation(self, input_path=None, output_dirpath=None, n=100, output_prefix='augmented_image', n_jobs=-1):
        
        if os.path.isfile(input_path):
            image_files = [input_path]
        elif os.path.isdir(input_path):
            image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and (not f.startswith('.'))]
        else:
            raise ValueError('Unknown types of this file: ' + input_path + '.')
        
        
        def __augmentation_ss(i):
            # randomly chose an image from the directory
            img_file = random.choice(image_files)
            img = skimage.io.imread(img_file)[:, :, :3]
            
            # randomly rotation
            img_ag = self.__augmentation_rotation(img)
            
            # fill up background (some case can not perform zero-padding)
            try:
                img_ag = self.__augmentation_fill_background(img_ag, img)
            except:
                print('image: ' + img_file)
            
            # randomly reflection
            img_ag = self.__augmentation_flip(img_ag)
            
            # randomly add noises
            img_ag = self.__augmentation_noise(img_ag)
            
            # resize
            img_ag = skimage.transform.resize(img_ag, (512, int(512 / img_ag.shape[0] * img_ag.shape[1])),
                                              anti_aliasing=False)
            img_ag = img_ag * 255
            img_ag = img_ag.astype(np.uint8)
            
            new_file_path = os.path.join(output_dirpath, output_prefix + '_' + str(i) + '.png')
            skimage.io.imsave(new_file_path, img_ag)
        
        
        r = joblib.Parallel(n_jobs=n_jobs, verbose=0)([joblib.delayed(__augmentation_ss)(i + 1) for i in range(n)])






