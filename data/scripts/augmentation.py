import os
import sys
import glob
from imgutils import imgUtils
import random
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





def augment(input_dpath, output_dpath, output_prefix, n_images):
    
    iu = imgUtils()
    
    
    for dpath in sorted(glob.glob(os.path.join(input_dpath, '*'))):
        print(dpath)
        random.seed(abs(hash(dpath)) % (10 ** 8))
        np.random.seed(abs(hash(dpath)) % (10 ** 8))
        
        dname = os.path.basename(dpath)
        _output_dpath = os.path.join(output_dpath, dname)
        if not os.path.exists(_output_dpath):
            os.makedirs(_output_dpath)
            
        if os.path.isdir(dpath):
            iu.augmentation(dpath, _output_dpath, n=n_images, output_prefix=output_prefix, n_jobs=-1)

  

   


if __name__ == '__main__':
    
    input_dpath = sys.argv[1]
    output_dpath = sys.argv[2]
    output_prefix = sys.argv[3]
    n_images = int(sys.argv[4])
    augment(input_dpath, output_dpath, output_prefix, n_images)




      


