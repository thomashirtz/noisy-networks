<!-- This is commented out. -->
# noisy-networks

This repository provide a minimal implementation of the [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf) using Pytorch

The actual implementation of this network is avalaible on my github repository [reinforcement-learning](https://github.com/thomashirtz/reinforcement-learning)


The noisy layers are similar to linear layer, but a noise that can be tuned with the "sigma" parameters is added.

<img src="https://render.githubusercontent.com/render/math?math=\Large y=w x%2Bb">

become:  
<img src="https://render.githubusercontent.com/render/math?math=\Large y=\left(\mu^{w}%2B\sigma^{w} \odot \varepsilon^{w}\right) x %2B \mu^{b}%2B\sigma^{b} \odot \varepsilon^{b}">
