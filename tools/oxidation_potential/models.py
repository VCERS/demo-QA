#!/usr/bin/python3

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GATv2Conv

class Predictor(nn.Module):
  def __init__(self, channels = 64, layer_num = 4, drop_rate = 0.2):
    super(Predictor,self).__init__()
    self.dense = nn.Linear(118, channels)
    self.convs = nn.ModuleList([GATv2Conv(channels, channels //8, 8, dropout = drop_rate) for _ in range(layer_num)])
    self.head = nn.Linear(channels, 1)
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    results = self.dense(x)
    for conv in self.convs:
      results = conv(results, edge_index)
    results = global_mean_pool(results, batch)
    results = self.head(results)
    results = torch.sinh(results)
    return results
