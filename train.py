import os
import sys
import argparse
from models import *


def train(class_labels, model_arch, model_inpath, model_outpath,
          traindata, validdata,
          epochs, batch_size, lr):
    
    dragonfly = DragonflyCls(model_arch=model_arch, input_size=(224, 224), model_path=model_inpath, class_labels=class_labels)
    
    dragonfly.train(traindata, validdata,
                    batch_size=batch_size, num_epochs=epochs, learning_rate=lr, save_best=False)
    
    dragonfly.save(model_outpath)
    
    
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Let dragonfly fly!')
   
    parser.add_argument('--class-label', required=True)
    parser.add_argument('--model-arch', required=True)
    parser.add_argument('--model-inpath', default=None)
    parser.add_argument('--model-outpath', default=None)
    parser.add_argument('-t', '--traindata', required=True)
    parser.add_argument('-v', '--validdata', required=True)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    args = parser.parse_args()
    
    train(args.class_label, args.model_arch, args.model_inpath, args.model_outpath,
          args.traindata, args.validdata,
          args.epochs, args.batch_size, args.lr)
    


