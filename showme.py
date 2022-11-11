import argparse
import training
import os
import json
import sys


# Para hacerlo interactivo por consola
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', default='./data',
                    type=str,
                    help='Indica al programa donde buscar el directorio de entrenamiento\
                          si la ruta no es especificada se buscará en el directorio /data')

parser.add_argument('-s', '--save', default='./models',
                    type=str,
                    help='Indica al programa donde buscar el directorio de entrenamiento  \
                          si se usa sin la badera -n se nombrará automaticamente al archivo \
                          model.pt')

parser.add_argument('-t', '--tags', default='./tags',
                    type=str,
                    help='Indica donde guardar las etiquetas de entrenamieto si no se da una \
                          ruta, las guarda en el directorio root por defecto, si se usa sin la badera \
                          -n se nombrará automaticamente al archivo tags.json')

parser.add_argument('-n', '--name', default=None,
                    type=str,
                    help='Indica el nombre que se le quiere dar al modelo entrenado a guardar, y sus etiquetas \
                         si se usa sin la bandera -s o -t, los guardara en la ruta por defecto con el nombre dado')

args = vars(parser.parse_args())

# Get args
data_dir = args['data']
model_save_path = args['save']
tags_save_path = args['tags']
model_name = args['name']

if not os.path.isdir(data_dir):
    print({f'{data_dir} no es un directorio'})

if not os.path.isdir(model_save_path):

    if model_save_path == './models':
        os.mkdir('./models')
    else:
        print({f'{model_save_path} no es un directorio'})
        sys.exit(-1)

if not os.path.isdir(tags_save_path):
    if tags_save_path == './tags':
        os.mkdir('./tags')
    else:
        print({f'{tags_save_path } no es un directorio'})
        sys.exit(-1)


if model_name:
    model_save_path += f'/{model_name}.pt'
    tags_save_path+= f'/{model_name}_tags.json'
else:
    model_save_path += '/model.pt'
    tags_save_path+= '/tags.json'


class_dir = f'{data_dir}/train'

classes = next(os.walk(class_dir))[1]
classes = sorted(classes)
tags = {}
print('Se encontraron las siguientes clases en el directorio de entrenamiento:')
print('id\tnombre')
for i, c in enumerate(classes, start=0):
    print(f'{i}\t{c}')
    tags[i] = c

try:
    n = len(classes)
    #training.get_trained_net(data_dir,n,num_epochs=8,device=training.ACTIVE_DEVICE,save_route=model_save_path)
except Exception:
    print('Ocurrio un error inesperado durante el entrenamiento, abortando..')
    sys.exit(-1)

json_object = json.dumps(tags, indent=0)
with open(tags_save_path, 'w', encoding='UTF8') as f:
    f.write(json_object)
sys.exit(0)