import os
import sys
import time
import logging
import coloredlogs
import glob
import copy
import geopy.distance
import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
import PIL
from PIL import Image
from PIL import ExifTags


logging.basicConfig(level = logging.INFO,
                    format = '[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt = '%Y-%m-%d %H:%M:%S')



class DragonflySqueezenet(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflySqueezenet, self).__init__()
        model = torchvision.models.squeezenet1_0(pretrained=True)
        self.base = model
        self.base.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.base.num_classes = n_classes
        
    def forward(self, x):
        x = self.base(x)
        return x



class DragonflyMobilenet(torch.nn.Module):
    
    def __init__(self, n_classes):
        super(DragonflyMobilenet, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)
        self.base = model
        
    
    def forward(self, x):
        x = self.base(x)
        return x
    


class DragonflyResnet(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflyResnet, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        self.base = model
    
    
    def forward(self, x):
        x = self.base(x)
        return x
    
    

class DragonflyVGG(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflyVGG, self).__init__()
        model =  torchvision.models.vgg11_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, n_classes)
        self.base = model
        
        
    def forward(self, x):
        x = self.base(x)
        return x



class DragonflyResnet152(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflyResnet152, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        self.base = model
    
    
    def forward(self, x):
        x = self.base(x)
        return x
    
    

class DragonflyVGG19(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflyVGG19, self).__init__()
        model =  torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, n_classes)
        self.base = model
        
        
    def forward(self, x):
        x = self.base(x)
        return x


class DragonflyDensenet(torch.nn.Module):

    def __init__(self, n_classes):
        super(DragonflyDensenet, self).__init__()
        model =  torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
        self.base = model

        
    def forward(self, x):
        x = self.base(x)
        return x










class nnTorchResize():
    
    def __init__(self, shape=(224, 224)):
        self.shape = shape


    def __call__(self, x):
        h, w, c = x.shape
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
        
        x = cv2.copyMakeBorder(x, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = cv2.resize(x, self.shape)
        x = PIL.Image.fromarray(x)
        
        return x





class nnTorchDataset(torch.utils.data.Dataset):

    def __init__(self, x, y=None, transforms=None):
        self.x = x
        self.y = y
        self.transforms = transforms
    
    
    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x = cv2.imread(self.x[i], cv2.IMREAD_COLOR)
        
        if self.transforms is not None:
            x = self.transforms(x)
        
        if self.y is None:
            return x
        
        else:
            y = self.y[i]
            return x, y
 





class DragonflyCls():

    
    def __init__(self, model_arch='vgg', input_size=(224, 224), model_path=None, class_labels=None, device=None):
        
        
        # set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # set model
        self.model_arch = model_arch
        self.input_size = input_size
        self.class_labels = self.__generate_labels(class_labels)
        self.model = self.__initialize_model(model_arch, model_path)
        self.model.to(self.device)
        
        logging.info('Model architecture:')
        logging.info('    input_size: {}; n_class:{}; model:{}; device: {}'.format(input_size, len(self.class_labels), str(model_arch), str(self.device)))
        if input_size[0] != 224 or input_size[1] != 224:
            logging.warning('The input_size was set as {}. Set to (224, 224), if you want to use imagenet pre-trained model.'.format(input_size))
        
        
        
        # set train log
        self.train_log = None
        
        
        # image normalization
        self.transforms = torchvision.transforms.Compose([
                nnTorchResize(self.input_size),
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomAffine(0.3, shear=0.3),
                torchvision.transforms.RandomRotation(degrees=45),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
        
        self.transforms_valid = torchvision.transforms.Compose([
                nnTorchResize(self.input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    
    
    
    def __generate_labels(self, class_labels_fpath):
        """
        Load 1 column file that contains all labels.
        """
        
        class_labels = []
        
        with open(class_labels_fpath, 'r') as infh:
            for class_name in infh:
                class_name = class_name.replace('\n', '')
                if class_name != '':
                    class_labels.append(class_name)
        
        return tuple(class_labels)
    
    
    
    
    
    def __initialize_model(self, model_arch=None, model_path=None):
        """
        Initialize a imagenet pre-trained model if the path to a model is not given,
        otherwise, load the pre-trained model.
        """
        
        model = None
        
        if model_arch == 'vgg':
            model = DragonflyVGG(len(self.class_labels))
        elif model_arch == 'vgg19':
            model = DragonflyVGG19(len(self.class_labels))
        elif model_arch == 'resnet':
            model = DragonflyResnet(len(self.class_labels))
        elif model_arch == 'resnet152':
            model = DragonflyResnet152(len(self.class_labels))
        elif model_arch == 'squeezenet':
            model = DragonflySqueezenet(len(self.class_labels))
        elif model_arch == 'mobilenet':
            model = DragonflyMobilenet(len(self.class_labels))
        elif model_arch == 'densenet':
            model = DragonflyDensenet(len(self.class_labels))
        
        
        if model_path is not None:
            if self.device == 'cuda':
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
            logging.info('Loaded the pre-trained model ({}).'.format(model_path))
        
        return model
    
    
    
    
    def __dataset_loader(self, dataset_path, load_mode=None, batch_size=32):
        """
        If the path is specified to a directory, load all images from the given directory.
        If the path is specified to a file, load the single image.
        """
        
        dataset = None
        if load_mode == 'train' or load_mode == 'valid':
            x = []
            y = []
            
            for i, class_label in enumerate(self.class_labels):
                for fpath in glob.glob(os.path.join(dataset_path, class_label, '*')):
                    if os.path.splitext(os.path.basename(fpath))[1].lower() in ['.jpg', '.jpeg', '.png']:
                        x.append(fpath)
                        y.append(i)
            
            if load_mode == 'train':
                dataset = nnTorchDataset(x, y=y, transforms=self.transforms)
            else:
                dataset = nnTorchDataset(x, y=y, transforms=self.transforms_valid)
                
            dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            logging.info('Loaded images from the directory {} for training.'.format(dataset_path))
        
        
        elif load_mode == 'inference':
            x = []
            y = []
            if os.path.isfile(dataset_path):
                x.append(dataset_path)
                y.append(dataset_path)
            else:
                for fpath in os.listdir(dataset_path):
                    if os.path.splitext(fpath)[1].lower() in ['.jpg', '.jpeg', '.png']:
                        x.append(os.path.join(dataset_path, fpath))
                        y.append(os.path.join(dataset_path, fpath))
                
            dataset = nnTorchDataset(x, y=y, transforms=self.transforms_valid)
            dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            logging.info('Loaded images from the directory {} for inference.'.format(dataset_path))
        
        else:
            raise ValueError('Only `train`, `valid` or `inference` can be specified.')
       
        
        return dataset


    




    def __train(self, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=50, save_best=True):
        since = time.time()
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
        
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                if phase == 'train':
                    lr_scheduler.step()
                
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if phase == 'train':
                    train_acc_history.append(epoch_acc.data.cpu().item())
                    train_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'valid':
                    val_acc_history.append(epoch_acc.data.cpu().item())
                    val_loss_history.append(epoch_loss)
    
        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        if save_best:
            self.model.load_state_dict(best_model_wts)
        
        self.train_history = {
            'train_acc': train_acc_history,
            'train_loss': train_loss_history,
            'val_acc': val_acc_history,
            'val_loss': val_loss_history
        }
    
    
    def gradcam(self, img_fpath):
        
        net = copy.deepcopy(self.model)
        net.eval()

        img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        if self.transforms_valid is not None:
            transforms = self.transforms_valid
            img = transforms(img)
        img = img.unsqueeze(0)
        
        def __extract_grad(grad):
            global feature_grad
            feature_grad = grad
        
        if self.model_arch == 'vgg19':
            x = net.base.features[:36](img)
            features = x
            features.register_hook(__extract_grad)
            x = net.base.features[36:](x)
            x = net.base.avgpool(x)
            x = x.view(x.size(0), -1)
            output = net.base.classifier(x)
            pred = torch.argmax(output).item()
       
        elif self.model_arch == 'resnet152':
            x = net.base.conv1(img)
            x = net.base.bn1(x)
            x = net.base.relu(x)
            x = net.base.maxpool(x)
            x = net.base.layer1(x)
            x = net.base.layer2(x)
            x = net.base.layer3(x)
            x = net.base.layer4(x)
            features = x
            features.register_hook(__extract_grad)
            x = net.base.avgpool(x)
            x = x.view(x.size(0), -1)
            output = net.base.fc(x)
            pred = torch.argmax(output).item()
            
        else:
            raise ValueError('Grad-CAM only support VGG19 or ResNet 152 archtecture!')
        
       
        # get the gradient of the output
        output[:, pred].backward()
        pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3])
        features = features.detach()
        for i in range(features.shape[1]):
            features[:, i, :, :] *= pooled_grad[i] 

        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.numpy()
        
        img = cv2.imread(img_fpath)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
        
        return superimposed_img

    
    def train(self, train_data_dpath, valid_data_dpath, batch_size=32, num_epochs=50, learning_rate=0.0001, save_best=True):
    
        # load dataset
        train_dataset = self.__dataset_loader(train_data_dpath, load_mode='train', batch_size=batch_size)
        valid_dataset = self.__dataset_loader(valid_data_dpath, load_mode='valid', batch_size=batch_size)
        
        dataloaders_dict = {'train': train_dataset, 'valid': valid_dataset}
    
        # train
        logging.info('The dragonfly is flying ... batch_size:{} epochs:{} lr:{}.'.format(batch_size, num_epochs, learning_rate))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        self.__train(dataloaders_dict, criterion, optimizer, lr_scheduler, num_epochs=num_epochs, save_best=save_best)
    
    
    
    
    def save(self, model_path):
        
        # save model
        torch.save(self.model.state_dict(), model_path)
        logging.info('The dragonfly is in a deep sleep at {}.'.format(model_path))
        train_history_path = os.path.splitext(model_path)[0] + '.train_hisotry.tsv'
        train_history_df = pd.DataFrame(self.train_history)
        train_history_df.to_csv(train_history_path, sep='\t', index=False)
        
    
    def inference(self, data_path):
        self.model.eval()
        dataloader = self.__dataset_loader(data_path, load_mode='inference')
        
        file_names = []
        pred_probs = None
        
        with torch.set_grad_enabled(False):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs).cpu().detach().numpy()
                file_names.extend(labels)
                if pred_probs is None:
                    pred_probs = outputs
                else:
                    pred_probs = np.concatenate([pred_probs, outputs], axis=0)
        
        pred_probs = pd.DataFrame(pred_probs, index=file_names, columns=self.class_labels)
        return pred_probs
        
        

class DragonflyMesh():
    
    def __init__(self, mesh):
        self.dragonflymesh = self.__load_meshdata(mesh)
    
    
    def __load_meshdata(self, mesh):
        x = pd.read_csv(mesh, header=0, sep='\t', index_col=0)
        x.index = x.index.map(str)
        
        dmesh = {
            'grid': x.iloc[:, :2],
            'mesh': x.iloc[:, 2:]
        }
        
        return dmesh
    
    
    def __dataset_loader(self, data_path):
        img_fpath = []
        if os.path.isfile(data_path):
            img_fpath.append(data_path)
        else:
            for fpath in os.listdir(data_path):
                if os.path.splitext(fpath)[1].lower() in ['.jpg', '.jpeg', '.png']:
                    img_fpath.append(os.path.join(data_path, fpath))
        
        return img_fpath
    
    
    def gis2mesh(self, lat, lng, order = 3):
        lat = float(lat)
        lng = float(lng)
        
        lat_in_min = lat * 60.0
        code12 = int(lat_in_min / 40)
        lat_rest_in_min = lat_in_min - code12 * 40
        code5 = int(lat_rest_in_min / 5 )
        lat_rest_in_min -= code5 * 5
        code7 = int(lat_rest_in_min / (5/10))

        code34 = int(lng) - 100
        lng_rest_in_deg = lng - int(lng)
        code6 = int(lng_rest_in_deg * 8)
        lng_rest_in_deg -= code6 / 8;
        code8 = int(lng_rest_in_deg / (1/80) )
        
        code = code12 * 100 + code34
        if order >= 2:
            code = code * 100 + code5 * 10 + code6
        if order == 3:
            code = code * 100 + code7 * 10 + code8
        
        return str(int(code))
    
     
    def get_jpeg_info(self, img_fpath):
        lat = None
        lng = None
        capture_date = None
        im = Image.open(img_fpath)

        exif = im._getexif()

        if exif is not None:
            exif = {ExifTags.TAGS[k]: v for k, v in exif.items() if k in ExifTags.TAGS}
            if 'GPSInfo' in exif:
                gps_tags = exif['GPSInfo']
                gps = {ExifTags.GPSTAGS.get(t, t): gps_tags[t] for t in gps_tags}
                is_lat = 'GPSLatitude' in gps
                is_lat_ref = 'GPSLatitudeRef' in gps
                is_lon = 'GPSLongitude' in gps
                is_lon_ref = 'GPSLongitudeRef' in gps

                if is_lat and is_lat_ref and is_lon and is_lon_ref:
                    lat = gps['GPSLatitude']
                    lat_ref = gps['GPSLatitudeRef']
                    if lat_ref == 'N':
                        lat_sign = 1.0
                    elif lat_ref == 'S':
                        lat_sign = -1.0
                    lon = gps['GPSLongitude']
                    lon_ref = gps['GPSLongitudeRef']
                    if lon_ref == 'E':
                        lon_sign = 1.0
                    elif lon_ref == 'W':
                        lon_sign = -1.0
                    lat = lat_sign * lat[0] + lat[1] / 60 + lat[2] / 3600
                    lng = lon_sign * lon[0] + lon[1] / 60 + lon[2] / 3600
        
            if 'DateTimeOriginal' in exif:
                capture_date = exif['DateTimeOriginal']
                capture_date = capture_date.split(' ')[0].replace(':', '-')
        return (capture_date, lat, lng)
    
    
    def __calc_dist(self, x):
        return geopy.distance.great_circle((x[0], x[1]), (x[2], x[3])).km
        
    
    def __predict(self, gis, d=100):
        gisdf =pd.DataFrame([gis] * self.dragonflymesh['grid'].shape[0],
                            index=self.dragonflymesh['grid'].index, columns=['lat0', 'lng0'])
        meshmat = pd.concat([self.dragonflymesh['grid'], gisdf], axis=1)
        in_range = (meshmat.apply(self.__calc_dist, axis=1) < d)
        output = self.dragonflymesh['mesh'].loc[in_range, :]
        output = output.sum(axis=0)
        output[output > 0] = 1
        if not all(in_range):
            output = output.fillna(1.0)
        
        return output
    
    
    
    def inference(self, data_path, d=100):
        dataset = self.__dataset_loader(data_path)
        pred_scores = None
        for img_fpath in dataset:
            capture_date, lat, lng = self.get_jpeg_info(img_fpath)
            if lat is not None and lng is not None:
                mesh = self.gis2mesh(lat, lng, 1)
                pred_score = self.__predict((lat, lng), d)
                pred_score = pd.DataFrame([pred_score.to_list()],
                                           index=[img_fpath], columns=self.dragonflymesh['mesh'].columns)
            else:
                pred_score = pd.DataFrame([[1.0 for i in range(self.dragonflymesh['mesh'].shape[1])]],
                                            index=[img_fpath], columns=self.dragonflymesh['mesh'].columns)
            
            if pred_scores is None:
                pred_scores = pred_score
            else:
                pred_scores = pd.concat([pred_scores, pred_score], axis=0)
        
        return pred_scores
        


