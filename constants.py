"""

Convenient constants

"""

import torch

DFLT_DATA_DIR = "./graphical_models/datasets/"
DFLT_MODEL_DIR = './inference/pretrained'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
