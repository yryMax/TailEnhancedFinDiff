### The Ultimate Goal
Sample p(x|y) given p(x) and y

e.g p(x) could be distribution generated, resampled, Gaussian sampled etc

e.g y could be momentum less than 0.0005

### The Preliminary Steps
According to bayes theory we sample p(x|y) by sampling from p(x)p(y|x).

Ideally we want hard requirement. specifically we want p(y|x) = 1 if x fulfill the conditions, otherwise p(y|x) = 0. 

But it is not very feasible because it doesn't have a gradient so that it doesnt provide any infomation on where to go
if the condition is not fulfilled 

So we have $p(y|x) = \exp(-s\cdot L(x))$ as an estimation

L(x) defines the panalty if the condition is not filled, s (guidance scale) controls the strength

For example if our condition is x<3 we can have L(x) = relu((3-x))^2

<img src="img_1.png" width="300">

another thinking approach 
$p(x) e^{-sL(x)} = \exp\Big( \log p(x) - sL(x) \Big) = \exp\Big( -s \cdot {\left[ L(x) - \frac{1}{s}\log p(x) \right]} \Big)$

Which is a Boltzmann distribution so s controls the inverse of the temperature and L controls the external energy

### The Traditional Approach

We can use rejection sampling to sample from $p^*(x) \propto p(x)\,e^{-sL(x)}$.

<img src="img_2.png" width="300">

Term correspondence

g(x) -> our p(x)

f(x) -> $\propto p(x)\,e^{-sL(x)}$

W -> $e^{-sL(x)}$

M -> 1, i.e. $\sup W$

And in the code it is one line `accept = np.random.rand() < np.exp(-guidance_scale * loss_i)`

BUT IT IS VERY SLOW WHEN THE CONDITION IS HARD TO FULFILL

### Gradient Guidance

Instead of sampling $p(x)$ and rejecting, we modify the reverse diffusion process so it directly samples from $p^*(x_0) \propto p(x_0)\,e^{-sL(x_0)}$.

We push $p^*$ forward to noise level $t$ to get $p^*(x_0)$ using the same forward kernel $q(x_t\mid x_0)$:

$$p^*_t(x_t) \;=\; \int q(x_t\mid x_0)\,p^*(x_0)\,dx_0 \;\propto\; \int q(x_t\mid x_0)\,p(x_0)\,e^{-sL(x_0)}\,dx_0$$

Bayes flip: $q(x_t\mid x_0)\,p(x_0) = p_t(x_t)\,p(x_0\mid x_t)$. Substituting:

$$p^*_t(x_t) \;\propto\; p_t(x_t)\cdot\underbrace{\mathbb{E}[e^{-sL(x_0)}\mid x_t]}_{=:\,E_t(x_t)}$$

Take log, then gradient

$$\nabla_{x_t}\log p^*_t(x_t) = \underbrace{\nabla_{x_t}\log p_t(x_t)}_{\text{model gives this}} + \underbrace{\nabla_{x_t}\log E_t(x_t)}_{\text{correction term}}$$

Approximate the posterior $p(x_0\mid x_t)$ by a point mass at $\hat x_0$

$$\log E_t(x_t) \;=\; \log\mathbb{E}[e^{-sL(x_0)}\mid x_t] \;\approx\; -s\,L(\hat x_0) \;\;\Longrightarrow\;\; \nabla_{x_t}\log E_t(x_t) \;\approx\; -s\,\nabla_{x_t}L(\hat x_0)$$


From score to posterior

$$p^*(x_{t-1}\mid x_t) \;\propto\; \underbrace{p(x_{t-1}\mid x_t)}_{\mathcal N(\mu,\Sigma)}\cdot E_{t-1}(x_{t-1})$$

Linearise $\log E_{t-1}$ around $x_{t-1}=\mu$:

$$\log E_{t-1}(x_{t-1}) \approx \log E_{t-1}(\mu) + (x_{t-1} - \mu)^\top\,g, \qquad g = \nabla_{x_{t-1}}\log E_{t-1}(x_{t-1})\big|_{\mu}$$


Gaussian × linear exponential = shifted Gaussian. Write $y = x_{t-1} - \mu$:

$$-\tfrac{1}{2}y^\top\Sigma^{-1}y \;+\; g^\top y \;=\; -\tfrac{1}{2}(y - \Sigma g)^\top\Sigma^{-1}(y - \Sigma g) + \text{const}$$

So the product is a Gaussian with shifted mean:

$$\mathcal N(\mu,\Sigma)\cdot e^{g^\top(x-\mu)} \;\propto\; \mathcal N(\mu + \Sigma g,\;\Sigma)$$

Plug in $g$

$$\boxed{\;\mu^* = \mu + \Sigma\cdot g = \mu - s\cdot\underbrace{\text{var}}_{\Sigma}\cdot\underbrace{\nabla_{x_t}L(\hat x_0)}_{\text{grad}}\;}$$

So also one line: `mean -= guidance_scale * var * grad` in `generate()`.

### Empirical Result



### Common Q&A

