<!-- This is commented out. -->
# noisy-networks

This repository provide a minimal implementation of the [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf) using Pytorch. The actual implementation of this network is avalaible on my [reinforcement-learning](https://github.com/thomashirtz/reinforcement-learning) github repository.

## Principle 
The noisy layers are similar to linear layer, but a noise that can be tuned with "sigma" parameters is added.

<img src="https://render.githubusercontent.com/render/math?math=\Large y=w x%2Bb">

become:  

<img src="https://render.githubusercontent.com/render/math?math=\Large y=\left(\mu^{w}%2B\sigma^{w} \odot \varepsilon^{w}\right) x %2B \mu^{b}%2B\sigma^{b} \odot \varepsilon^{b}">

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

In the case of the Independent version, be careful to not input a tuple into the `torch.FloatTensor(features)`, otherwise it will create a tensor with those values. Instead, it is possible to unpack them `torch.FloatTensor(*features)`  



