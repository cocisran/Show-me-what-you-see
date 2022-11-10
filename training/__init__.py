import torch

ACTIVE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from .transfernet import transfernet_squeeznet, preprocess, initialize_model

get_trained_net = transfernet_squeeznet

