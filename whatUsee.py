import torch
import cv2
import argparse
import time
import training
import numpy as np
import json
import sys


#Para hacerlo interactivo por consola
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default=None, 
                    help='path to the input image')

parser.add_argument('-d', '--device', default='cpu', 
                    help='computation device to use', 
                    choices=['cpu', 'cuda'])

parser.add_argument('-m', '--model', default='models/model.pt', 
                    help='path al modelo, valor por defecto: \
                          models/model.pt')


parser.add_argument('-t', '--tags', default='tags/tags.json', 
                    help= 'path a las eiquetas del modelo, valor del modelo:\
                         tags/tags.json')

args = vars(parser.parse_args())

# Variables de ejecución
DEVICE = args['device']
MODEL = args['model']
TAGS = args['tags']
image = cv2.imread(args['input'])

#Obtener tags de clasficacion
tags:dict = {}

with open(TAGS, 'r') as openfile:
    tags = json.load(openfile)
 

if tags is not {}:
    num_classes = len(tags.keys())
else:
    print('No se encontraron las etiquetas, abortando...')
    sys.exit(-1)


# carga el modelo
try:
    model, input_size = training.initialize_model(num_classes)
    if args['model'] is not None:
        model.load_state_dict(torch.load(args['model']))
except Exception as e:
    print(e)
    print('No se pudo cargar el modelo con exito, abortando')
    sys.exit(-1)

model.eval()
transform = training.preprocess()

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(DEVICE)
model.to(DEVICE)

with torch.no_grad():
    start_time = time.time()
    output = model(input_batch)
    end_time = time.time()

probabilities = torch.nn.functional.softmax(output[0], dim=0)
probabilities = probabilities.numpy()

print("Probabilidad por clase")
for i in range(len(probabilities)):
    print(f'{tags[str(i)]} : {probabilities[i]}')

print(f"Tiempo de cálculo: {(end_time-start_time):.3f} seconds")