import torch
import torch.nn as nn
import model.CSModel as CSModel
PATH = ''
model = CSModel()
model.load_state_dict(torch.load(PATH))