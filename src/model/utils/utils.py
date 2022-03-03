# author: 
# contact: ycfrude@163.com
# datetime:2022/3/2 2:45 PM
import torch
from typing import List

def init_weight(weight_shape: List[int]):
    weight = torch.empty(weight_shape)
    torch.nn.init.xavier_normal_(weight, gain=1)
    return weight