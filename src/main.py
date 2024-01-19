
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import time
import dlib
import matplotlib.colors as mcolors	
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets
import torch_dct as dct
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import cv2
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
from torch.optim import lr_scheduler
classes_num = 8
from ECANet.ECA import eca_resnet101
cuda_index = 3


device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')

net = eca_resnet101(num_classes=8)
net.load_state_dict(torch.load(" "),
                             strict=False)

net1 = models.resnet18(weights=None) 
net1.load_state_dict(torch.load(" "),
                             strict=False)

def fine_class_loss(output, targets):


 
    classes = ['amusement', 'anger', 'awe', 'contentment',  'disgust', 'excitement','fear','sadness'] 
    fine_classes = {'group1': ['fear', 'amusement', 'anger','disgust'], 'group2': ['awe', 'contentment', 'excitement','sadness']}

 
    class_to_group = {c: i for i, group in enumerate(fine_classes.values()) for c in group}


    class_to_idx = {c: i for i, c in enumerate(classes)}



    def map_to_major_classes(labels):
        return torch.tensor([class_to_group[classes[l]] for l in labels]).to(device)

    major_labels = map_to_major_classes(targets).to(device)

    
    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, major_labels)

    return loss

 
def gram_matrix(tensor):

    if len(tensor.shape) != 4:
        raise ValueError("输入张量的维度必须为 [b, c, h, w]。")

    b, c, h, w = tensor.shape

    transposed_tensor = tensor.permute(0, 1, 3, 2)

    product = torch.matmul(tensor, transposed_tensor)

    return product      


class CompleteNetwork(torch.nn.Module):
    def __init__(self):
        super(CompleteNetwork, self).__init__()

        self.resnet_features = torch.nn.Sequential(*list(net.children())[:-2])
        self.face_features = torch.nn.Sequential(*list(net1.children())[:-2])
        self.resnet1 = torch.nn.Sequential(*list(net.children())[:-4]) # 512 28 28
        self.resnet2 = torch.nn.Sequential(*list(net.children())[:-5]) # 256 56 56
        self.resnet3 = torch.nn.Sequential(*list(net.children())[:-7]) # 64 112 112
        self.Ada =  nn.AdaptiveAvgPool2d((1,1))
        self.cls = nn.Linear(3392,classes_num)
    
    def forward(self, image,face_tensor):
    

        x = self.resnet_features(image)  #feature 2048 7 7
        x1 = self.face_features(face_tensor)  #feature 512 2 2
        x512 = self.resnet1(image) #feature 512 28 28
        x256 = self.resnet2(image) #feature 256 56 56 
        x64 = self.resnet3(image) #feature 64 112 112
        
        output0 = self.Ada(x)
        output1 = self.Ada(x1)
        gram1  = gram_matrix(x512)
        gram2  = gram_matrix(x256)
        gram3  = gram_matrix(x64)
        #print(f"gram1 dim is {gram1.shape}")
        input2 = self.Ada(gram1)
        input3 = self.Ada(gram2)
        input4 = self.Ada(gram3)
       
        output = torch.cat((output0,output1,input2,input3,input4),dim=1)
        output = output.squeeze()
        output = self.cls(output)

        return output

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    
    def __getitem__(self, index):
        # ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0] 
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    

if __name__ == '__main__':
    device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')
    now = datetime.datetime.now()
    message = ''
    print(now.strftime("%Y-%m-%d %H:%M:%S")) 
  
    print(message) 
    print("using {} device.".format(device))
   
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),


        "valid": transforms.Compose([transforms.Resize(256),  
                                     transforms.CenterCrop(224),  
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  
    image_path = os.path.join(data_root, "")  
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path) 
   
    
    train_dataset = ImageFolderWithPaths(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=24)

    validate_dataset = ImageFolderWithPaths(root=os.path.join(image_path, "valid"),
                                            transform=data_transform["valid"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=24)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                         val_num))
                                                                         

    model = CompleteNetwork()
   

    model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer =  optim.SGD((param for param in model.parameters()), lr=0.0001, momentum=0.9,weight_decay=0.0005)

    scheduler = CosineAnnealingLR(optimizer, T_max=90, eta_min=0.0000000001)
    a = 0.3
    epochs = 60
    best_acc = 0.0
    save_path = ''
    running_acc = 0
    train_steps = len(train_loader)
    val_acc= []
    train_loss = []
    all_predictions = []
    all_labels = []
    class_correct = [0.] * 10
    class_total = [0.] * 10

    y_test, y_pred = [], []
    train_accuracy = []

    detector = dlib.get_frontal_face_detector()
    fit_time = time.time()
    for epoch in range(epochs):

        # train
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels, paths = data
            images = images.to(device)
            labels = labels.to(device)
            result_tensor = torch.empty(0, 3, 48, 48)

            for i in range(len(paths)):
                image = cv2.imread(paths[i])
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
                faces = detector(gray_img)
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
                    x, y, w, h = (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())

                    largest_face_image = image[y:y+h, x:x+w]

                    if largest_face_image.size > 0:
                        resized_face = cv2.resize(largest_face_image, (48, 48))
                
                        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                        resized_face = np.transpose(resized_face, (2, 0, 1))
                        result_tensor = torch.cat((result_tensor, torch.Tensor(resized_face).unsqueeze(0).float()), dim=0)
                    else:
                        resized_face = torch.zeros(1, 3, 48, 48)
                        result_tensor = torch.cat((result_tensor, resized_face), dim=0)
                else:
                    resized_face = torch.zeros(1, 3, 48, 48)
                    result_tensor = torch.cat((result_tensor, resized_face), dim=0)

            optimizer.zero_grad()
            logits = model(images,result_tensor.to(device))
          
            loss1 = loss_function(logits, labels)
            loss2 = fine_class_loss(logits,labels)
            pred_train = torch.max(logits, dim=1)[1]
            loss = loss1 + a*loss2
            loss.backward()
            optimizer.step()


            
            running_loss += loss.item()  
            running_acc += torch.eq(pred_train, labels).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            train_accuracy.append(running_acc)
            train_loss.append(loss.item())
       

        # validate
        model.eval()
        acc = 0.0  
      
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for step, data in enumerate(val_bar):
                images, val_labels, paths = data
                images = images.to(device)
                val_labels = val_labels.to(device)
                result_tensor = torch.empty(0, 3, 48, 48)

                for i in range(len(paths)):
                    image = cv2.imread(paths[i])
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    faces = detector(gray_img)
                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
                        x, y, w, h = (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())

                        largest_face_image = image[y:y+h, x:x+w]

                        if largest_face_image.size > 0:
                            resized_face = cv2.resize(largest_face_image, (48, 48))
                   
                            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                            resized_face = np.transpose(resized_face, (2, 0, 1))
                            result_tensor = torch.cat((result_tensor, torch.Tensor(resized_face).unsqueeze(0).float()), dim=0)
                        else:
                            resized_face = torch.zeros(1, 3, 48, 48)
                            result_tensor = torch.cat((result_tensor, resized_face), dim=0)
                    else:
                        resized_face = torch.zeros(1, 3, 48, 48)
                        result_tensor = torch.cat((result_tensor, resized_face), dim=0)
                outputs = model(images,result_tensor.to(device))
              
                predict_y = torch.max(outputs, dim=1)[1]
    
                acc += torch.eq(predict_y, val_labels).sum().item()
                all_predictions.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
              
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                val_acc.append(acc)
                
                
                predicted = predict_y.cpu()
                c = (predicted == val_labels.cpu()).squeeze()
                for i, label in enumerate(val_labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        train_acc = running_acc/ train_num 

        val_accurate = acc / val_num
        scheduler.step()

        print('[epoch %d] train_loss: %.3f   train_acc:%.4f val_accuracy: %.4f' %
              (epoch + 1, running_loss / len(train_loader), train_acc, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    if classes_num == 6:
            classes = ['anger', 'joy', 'surprise', 'disgust',  'fear', 'sadness']  
    else:
            classes = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness'] 
    for i in range(len(classes)):
        class_acc = 100 * class_correct[i] / max(1, class_total[i]) 
        print(f'Accuracy of {classes[i]} : {class_acc:.2f}%') 
    print('Finished Training')
    print('Total time: {:.2f} hours'.format((time.time() - fit_time) / 3600))
    print(message)
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


