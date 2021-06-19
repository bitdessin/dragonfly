import os
import sys
import glob
import random
import shutil
import numpy as np
import cv2





def make_mask(f_input, f_output):
    
    img = cv2.imread(f_input)
    b_ch, g_ch, r_ch = cv2.split(img)
    
    img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 9, 17)
    img_dragonfly = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_dragonfly = cv2.adaptiveThreshold(img_dragonfly, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 25, 9)
    
    kernel = np.ones((2, 2), np.uint8)
    img_dragonfly = cv2.dilate(img_dragonfly, kernel, iterations=3)
    
    img_bgra = cv2.merge((b_ch, g_ch, r_ch, img_dragonfly))
    cv2.imwrite(f_output, img_bgra)




if __name__ == '__main__':
    
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for d in sorted(glob.glob(os.path.join(data_path, '*'))):
        d_output_path = os.path.join(output_path, os.path.basename(d))
        
        if not os.path.exists(d_output_path):
            os.mkdir(d_output_path)
        
        for f in glob.glob(os.path.join(d, '*')):
            f_output_path = os.path.join(d_output_path, 'mask_' + os.path.basename(f))
            f_output_path = os.path.splitext(f_output_path)[0] + '.png'
            make_mask(f, f_output_path)
            
            
            
    
    
    

