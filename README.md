# Dragonfly Classification

## Usage

### Preparation

The pre-trained weights for species or genus classification models can be downloaded with the following scripts. File names ending with `_resnet152.pth` and `_vgg19.pth` are the weights of PyTorch models of ResNet152 and VGG19, respectively. File names starting with `meshmatrix_` are the summary of ecological survey data of dragonflies and damselflies.

```bash
mkdir weights
wget -P ./weights https://biunit.org/storage/dragonfly/species_resnet152.pth
wget -P ./weights https://biunit.org/storage/dragonfly/species_vgg19.pth
wget -P ./weights https://biunit.org/storage/dragonfly/genus_resnet152.pth
wget -P ./weights https://biunit.org/storage/dragonfly/genus_vgg19.pth
wget -P ./weights https://biunit.org/storage/dragonfly/meshmatrix_species.tsv.gz
wget -P ./weights https://biunit.org/storage/dragonfly/meshmatrix_genus.tsv.gz
```


### Species Identification

To predict species name of dragonflies and damselflies with ResNet152 model, run the following scripts. Note that it can use VGG19 model for prediction by changing `resnet152` to `vgg19` in the scripts. The prediction result will be saved in `inf_probs.txt` specified with `-o` option. 

```bash
python predict.py --class-label  classes_species.txt             \
                  --model-arch   resnet152                       \
                  --model-weight ./weights/species_resnet152.pth \
                  -i data/dataset_T/example_01.jpg               \
                  -o inf_probs.txt
```

To perform the prediction with combined model (i.e., image model and additional ecological filtering), add `--mesh` option and run the following scripts.

```bash
python predict.py --class-label  classes_species.txt                 \
                  --model-arch   resnet152                           \
                  --model-weight ./weights/species_resnet152.pth     \
                  --mesh         ./weights/meshmatrix_species.tsv.gz \
                  -i data/dataset_T/example_01.jpg                   \
                  -o inf_probs.txt
```


### Genus Identification

To predict genus of dragonflies and damselflies with image models, run the following scripts with the model weight for the genus level (e.g., `genus_resnet152.pth`).

```bash
python predict.py --class-label  classes_genus.txt             \
                  --model-arch   resnet152                     \
                  --model-weight ./weights/genus_resnet152.pth \
                  -i data/dataset_T/example_01.jpg             \
                  -o inf_probs.txt
```

To use the combined model, add `--mesh` option and run the following scripts.

```bash
python predict.py --class-label  classes_genus.txt                 \
                  --model-arch   resnet152                         \
                  --model-weight ./weights/genus_resnet152.pth     \
                  --mesh         ./weights/meshmatrix_genus.tsv.gz \
                  -i data/dataset_T/example_01.jpg                 \
                  -o inf_probs.txt
```





### Training

```bash
python train.py --class-label   classes_species.txt                 \
                --model-arch    resnet152                           \
                --model-outpath ./weights/example_model.pth         \
                --traindata     ./data/dataset_W1/augmentated_image \
                --validdata     ./data/dataset_F/raw                \
                --epochs 5 --batch-size 32 --lr 0.001
```

