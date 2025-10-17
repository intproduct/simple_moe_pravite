# Expert layers for Mixture of Experts (MoE) models.

import mlx.core as mx
import mlx.nn as nn
from typing import List, Type, Any, Optional


class MLPExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=2, activation:Type[nn.Module]=nn.ReLU):
        super().__init__()
        layer_size = [input_dim] + [hidden_dim] * (num_layers) + [output_dim]
        self.layers = [
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(layer_size[:-1], layer_size[1:])
        ]
        self.activation = activation()

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
    

class CNNExpert(nn.Module):
    def __init__(self, input_channels: int, num_filters:List[int], kernel_size:int, output_dim:int, image_size:int, activation:Type[nn.Module]=nn.ReLU):
        super().__init__()
        self.num_layers = len(num_filters)
        self.image_size = image_size
        self.activation = activation()

        self.layers = []

        cerrent_channel = input_channels

        for i in range(self.num_layers):
            out_channel = num_filters[i]
            paddding = (kernel_size - 1) // 2

            self.layers.append(
                nn.Conv2d(cerrent_channel, out_channel, kernel_size, padding=paddding),
            )
            self.layers.append(self.activation())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_channel = out_channel

        lp_input = mx.zeros((1, image_size, image_size, input_channels))

        lp_output = self._forward(lp_input)

        flattened_size = lp_output.size

        self.out = nn.Linear(flattened_size, output_dim)   

    def _forward(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self._forward(x)
        x = x.reshape((x.shape[0], -1))
        x = self.out(x)
        return x
    
class RNNExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=2, rnn_type: str='LSTM', activation:Type[nn.Module]=nn.ReLU):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = activation()

    def __call__(self, x: mx.array) -> mx.array:
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.activation(self.fc(out))
        return out 
