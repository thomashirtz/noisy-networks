<!-- This is commented out. -->
# noisy-networks

This repository provide a minimal implementation of the paper [[Noisy Networks for Exploration]](https://arxiv.org/pdf/1706.10295.pdf) using Pytorch. Actual utilization of this network on reinforcement learning algorithms are avalaible on my [[reinforcement-learning]](https://github.com/thomashirtz/reinforcement-learning) github repository.

## Principle 
The noisy layers are similar to linear layer, but a noise that can be tuned with "sigma" parameters is added.

<img src="https://render.githubusercontent.com/render/math?math=\Large y=w x%2Bb">

Become:  

<img src="https://render.githubusercontent.com/render/math?math=\Large y=\left(\mu^{w}%2B\sigma^{w} \odot \varepsilon^{w}\right) x %2B \mu^{b}%2B\sigma^{b} \odot \varepsilon^{b}">

## DQN Implemetation example

```
class DQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc_1 = nn.Linear(input_features, HIDDEN_UNITS)
        self.fc_2 = nn.Linear(HIDDEN_UNITS, output_features)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        return self.fc_2(x)
```

Become:  

```
from noisynetworks import FactorisedNoisyLayer

class DQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.noisylayer_1 = FactorisedNoisyLayer(input_features, HIDDEN_UNITS)
        self.noisylayer_2 = FactorisedNoisyLayer(HIDDEN_UNITS, output_features)

    def forward(self, x):
        x = F.relu(self.noisylayer_1(x))
        return self.noisylayer_2(x)
```
For replay function that works with batch, the rest of the code is almost unchanged, since by default, everytime the forward loop is called the noise will change. The other exploration techniques such as epsilon-greedy can be removed.

## Implementation details

`nn.Parameter()` is indispensable when using tensor as custom parameter, otherwise, the optimizer will not know that they exist.  
`self.register_buffer()` in the initialization allows to link those parameters to the layer, without setting them as trainable.  
`F.linear(x, weight=self.mu_weight, bias=self.mu_bias)` allows to elegantly do the linear computation.  
`torch.ger(a, b)` allows to elegantly create a 2D tensor with two 1D tensors.  
`tensor.uniform_(a, b)` allows to fill a tensor with an uniform distribution bounded by a and b.  
`tensor.fill_(x)` allows to fill a tensor with x.  
Adding the `data` in the lines such as `self.sigma_bias.data.fill_(self.sigma)` allows to not have in place modification errors.  
In the forward loop, there is the possibility to either manually sample the noise by setting `sample_noise` to `False`.  
```
if sample_noise:
    self.sample_noise()
```

When the `self.training` is set to `False`, it is possible to do a forward pass without the noise:  
```
if not self.training:
    return F.linear(x, weight=self.mu_weight, bias=self.mu_bias)
```

In the case of the Independent version, be careful to not input a tuple into the `torch.FloatTensor(features)`, otherwise it will create a tensor with those values. Instead, it is possible to unpack them `torch.FloatTensor(*features)`.  

# Installation
Direct Installation from github using pip by running this command:
```
pip install git+https://github.com/thomashirtz/noisy-networks#egg=noisynetworks
```

