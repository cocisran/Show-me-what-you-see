from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import copy

# Codigo extraido del tutorial de Nathan Inkawhich
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def train_model(model, dataloaders, criterion, optimizer, num_epochs=15,device="cuda"):
    '''
    Maneja el entrenamiento y la validacion del modelo pasado por parametro
    :model:         modelo de red a entrenar
    :dataloaders:   diccionario con los dataset de entrenamiento
    :criterion:     funcion de error 
    :optimizer:     optimizador de pesos
    :num_epochs:    número de epocas de entrenamiento
    :device:        procesador usado, gpu o cuda
    '''
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Entrenamiento completado en  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Mayor valor de precision: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model):
    '''
    Esta funcion "Congela los pesos" de la red actual, para
    que no sean modificados durante el entrenamiento
    '''
    for param in model.parameters():
        param.requires_grad = False


def initialize_model(num_classes:int):
    '''
    Ajusta el modelo pre-entrenado a nuestras necesidades actuales
    '''
    model_ft = None
    input_size = 0

    model_ft = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
    set_parameter_requires_grad(model_ft)
    #Cambia la capa de salida
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    input_size = 224



    return model_ft, input_size

def preprocess():
    """
    Prepara una imagen para ajustarse a la entrada de la red 224 x 224 x 3
    (imagenes de 224 x 224 de 3 canales de color RGB)
    Y normalizarla a los valores pedidos en la documentación
    """
    transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),])
    return transform