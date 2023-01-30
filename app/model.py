from torch import nn
from collections import OrderedDict

class CSModel(nn.Module):
    def __init__(self) -> None:
        super(CSModel, self).__init__()
        self.model = nn.Sequential(OrderedDict({'Linear' : nn.Linear(768, 768),
                                        'Active_fn' : nn.ReLU(),
                                        'Dropout' : nn.Dropout(p=0.1),
                                        'cls_layer' : nn.Linear(768, 2)}))
    def forward(self,input):
        output = self.model(input)
        return output
