#coding=utf8
import math
import torch
import torch.nn as nn
import dgl.function as fn
from model.model_utils import Registrable, FFN
from model.encoder.functions import *

@Registrable.register('hansql')
class HANSQL(nn.Module):
    def __init__(self, args):
        super(HANSQL, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.num_heads = args.num_heads
        self.ndim = args.gnn_hidden_size

    def forward(self, x, batch):
        qx = x.masked_select(batch.graph.question_mask.unsqueeze(-1)).view(-1, self.ndim)
        tx = x.masked_select(batch.graph.table_mask.unsqueeze(-1)).view(-1, self.ndim)
        cx = x.masked_select(batch.graph.column_mask.unsqueeze(-1)).view(-1, self.ndim)
        return x
