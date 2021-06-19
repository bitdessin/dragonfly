# Image Augmentation


```bash
cd dataset_W1
python ../scripts/augmentation.py raw augmentated_image augment 5
cd ..

cd dataset_W2
python ../scripts/make_dragonfly_mask.py raw mask
python ../scripts/make_dragonfly_synthesis.py mask ../background synthesis 5
cd ..

cd dataset_F
python ../scripts/augmentation.py raw augmentated_image augment 5
cd ..
```

