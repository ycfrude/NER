# author: 
# contact: ycfrude@163.com
# datetime:2022/3/1 3:54 PM

from typing import List, Union, Dict
from collections import OrderedDict
import torch
import torch.nn as nn

_activate_func = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}
_base_type = {int, float, dict, list, tuple, str}


class IDCNN(nn.Module):
    '''
        对论文的cnn部分的复现
        [ITERATED DILATED CONVOLUTIONAL NEURAL NETWORKS FOR WORD SEGMENTATION](https://sceweb.sce.uhcl.edu/xiaokun/doc/Publication/2021/NNW2021_HHe.pdf)
        在原论文的基础上增加了 skip connect(sum) 与 augment skip connect(aug)
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None, kernel_size: int = 3,
                 num_block=1, max_length=32, layers: List[Union[int, Dict[str, int]]] = None,
                 hidden_act="relu", residual_mode=None):
        super(IDCNN, self).__init__()
        if layers is None:
            layers = [1, 2, 2]
        if output_size is None:
            output_size = hidden_size
        self._set_attributes(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             kernel_size=kernel_size, num_block=num_block, max_length=max_length,
                             layers=layers, hidden_act=hidden_act, residual_mode=residual_mode)
        self.linear = nn.Linear(input_size, hidden_size)
        self.idcnn = nn.ModuleDict()
        for i in range(num_block):
            if i == 0:
                in_channels = input_size
            elif residual_mode == "aug":
                in_channels = 2 * hidden_size
            else:
                in_channels = hidden_size
            self.idcnn[f"idcnn_block_{i}"] = \
                IDCNNBlock(in_channels, hidden_size, kernel_size, max_length, layers,
                           hidden_act, residual_mode)
        if residual_mode == "aug":
            in_channels = 2 * hidden_size
        else:
            in_channels = hidden_size
        self.pooler = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_channels, output_size)),
            ("activate_function", _activate_func[hidden_act]())
        ]))

    def _set_attributes(self, **kawagrs):
        for k, v in kawagrs.items():
            self.__setattr__(k, v)

    def forward(self, inputs, mask=None):
        # output = inputs.permute(0, 2, 1)
        output = inputs
        for k, v in self.idcnn.items():
            output = v(output, mask)
            if self.residual_mode == "sum":
                output = inputs + output
            elif self.residual_mode == "aug":
                output = torch.cat([inputs, output] , dim=-1)
        # output = output.permute(0, 2, 1)
        return self.pooler(output)

    @property
    def config(self):
        config = {k: v for k, v in self.__dict__.items() if type(v) in _base_type}
        return config


class IDCNNBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size, kernel_size: int=3,
                 max_length=32, layers: List[int] = None, hidden_act="relu",
                 residual_mode: str=None
                 ):
        super(IDCNNBlock, self).__init__()
        idcnn_block = nn.ModuleDict()
        parameter_weight = IDCNNBlock.init_param([input_size, hidden_size, kernel_size])
        for i, layer in enumerate(layers):
            idcnn_block[f"idcnn_layer_{i}"] = \
                IDCNNLayer(hidden_size, hidden_size, kernel_size, max_length, layer,
                           hidden_act, parameter_weight)
        self.idcnn_block = idcnn_block
        self.residual_mode = residual_mode

    @staticmethod
    def init_param(weight_shape:List[int]):
        weight = torch.empty(weight_shape)
        torch.nn.init.xavier_normal_(weight, gain=1)
        return nn.Parameter(weight)

    def forward(self, inputs, mask=None):
        output = inputs
        for i, (k, v) in enumerate(self.idcnn_block.items()):
            output = v(inputs=output, mask=mask)
        return output


class IDCNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3,
                 max_length=32, dilation=1, hidden_act="relu",
                 parameter_weight=None):
        super(IDCNNLayer, self).__init__()
        idcnn_layer = nn.Sequential()
        conv = Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      dilation=dilation,
                      padding=kernel_size // 2 + dilation - 1,
                      parameter_weight=parameter_weight)
        idcnn_layer.add_module(f"conv", conv)
        idcnn_layer.add_module("activate_function", _activate_func[hidden_act]())
        idcnn_layer.add_module("layernorm", LayerNorm(max_length))
        self.idcnn_layer = idcnn_layer

    def forward(self, inputs: torch.tensor, mask=None):
        if mask is not None:
            inputs[torch.eq(mask, 0)] = 0
        inputs = inputs.permute(0, 2, 1)
        output = self.idcnn_layer(inputs)
        output = output + inputs
        return output.permute(0, 2, 1)


class Conv1d(nn.Conv1d):
    '''
    支持 共享参数
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 parameter_weight=None):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                     padding_mode=padding_mode)
        if parameter_weight is not None:
            self.weight = parameter_weight


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

