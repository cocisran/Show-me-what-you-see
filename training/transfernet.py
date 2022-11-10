from __future__ import print_function
from __future__ import division
from .utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# Codigo extraido del tutorial de Nathan Inkawhich
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def transfernet_squeeznet(data_dir:str
                         ,num_classes:int,
                         batch_size:int = 8,
                         num_epochs:int = 15,
                         device:str = 'cpu',
                         save_route:str = None):
    '''
    Transfiere el conocimiento de la red Squeeznet para nuestros propositos personales. El numero de clases debe 
    coincidir con el número data sets de clases otorgados para reentrenar la red.
    
    :data_dir:      PATH a donde se encuentra los data sets, se asumen que tienen el formato 
                    train[clase1 ... claseN] val[clase1 ... claseN]
    :num_classes:   número de clases a identificar (sálida de la ultima capa de la red)
    :batch_size:    tamaño del bloque de datos usado para el entrenamiento
    :num_epochs:    número de rondas de entrenamiento
    :device:        dispositivo usado por el entrenamiento, cpu o cuda
    :save_route:    cuando se usa, guarda el modelo resultado del entrenamiento de manera persistente.
    '''

    #Obtener el modelo preentrenado de squeeznet
    model_ft, input_size = initialize_model(num_classes)

    # Estableciendo las transformaciones de entrenamiento y validación
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    } 
    # Creando datasets de entrenamiento
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Creando datasets de validacion
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Mueve el modelo al dispositivo a usar
    model_ft = model_ft.to(device)

    # Selecciona los parametros a ajustar
    params_to_update = model_ft.parameters()
    print("Params to learn:")

    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Se establece el optimizador para los parametros a ajustar
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Funcion de error 
    criterion = nn.CrossEntropyLoss()

    # Iniciar entrenamiento 
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs ,device=device)

    # Guardar modelo
    if save_route is not None:
        torch.save(model_ft.state_dict(), save_route)
    
    return model_ft