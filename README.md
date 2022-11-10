# Show me what you see
 
 Es un paquete de python3 que incluye dos scritps para el entrenamiento y uso de redes neuronales, en tareas de reconocimiento visual.

## Preparar el ambiente

Este paquete usa python >= 3.10.6
```bash
python -m venv .env
source .env/bin/activate
pip3 install -r requeriments.txt
```

## showme.py
Este módulo de python se encarga de transferir el conocimiento de la red [squeeznet](https://pytorch.org/hub/pytorch_vision_squeezenet/) a un nuevo modelo, entrenado en base a un nuevo dataset proporcionado. 

## whatUsee.py  :construction_worker_man:

*Aún en desarrollo*

Este módulo tomará una red ya entrenada a la cual podremos alimentar y nos dirá cual es la clase a la que con mayor probabiliad pertence.

