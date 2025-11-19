---
title: "The Principles of Diffusion Models"
parent: Notes
layout: default
---

# Table of Contents

> Links: [arXiv](https://www.arxiv.org/abs/2510.21890)  
> This summary is based on version 1 of the paper. (https://www.arxiv.org/abs/2510.21890v1)  


Part A & B. Foundations of Diffusion Models
Part A. Introduction to Deep Generative Models
  1. Deep Generative Modeling
    1.1. What is Deep Generative Modeling?
    1.2. Prominent Deep Generative Models
    1.3. Taxonomy of Modelings
    1.4. Closing Remarks
B. Core Perspectives on Diffusion Models
  2. Variational Perspective
    2.1. Variational Autoencoder
    2.2. Variational Perspective: DDPM
    2.3. Closing Remarks



Part C & D. Controlling and Accelerating the Diffusion Sampling  
Part C. Sampling of Diffusion Models  
Part D. Learning Fast Generative Models  


---

<img width="90%" alt="Part AB Overview" src="https://github.com/user-attachments/assets/a67fa537-8467-4f2e-a361-a7564cd84732">



# 1. Deep Generative Modeling

> "*What I cannot create, I do not understand*" by Richard P. Feynman


## 1.1. What is Deep Generative Modeling?

**What is Deep Generative Models (DGMs)?**  
DGMs are neural networks that learn a probability distribution to generate new samples that resemble the dataset.  

**How to train DGMs?**  
Fit the model parameter $$\phi$$ by minimizing a discrepancy that measures how far the model distribution $$p_{\phi}$$ is from the data distribution $$p_{data}$$.  

**How to generate samples using DGMs?**  
Run the model's sampling procedure to draw $$\mathbf{x} \sim p_{\phi}$$.  

**How to measure the quality of DGMs?**  
Quality of generated samples & summary statistics between generated samples and dataset.  


### 1.1.1. Mathematical Setup

> Assumption: a finite set of samples is drawn independently and identically distributed (i.i.d.) from the data distribution $$p_{data}$$  

$$\phi \in \underset{\phi}{\mathrm{argmin}} \ D(p_{data}, p_{\phi})$$

2 main families of $$D$$: *f-divergences*, *Optimal Transport (OT) distances*  

**Forward Kullback-Leibler (KL) divergence**  
Mode covering, minimizing forward KL = Maximum Likelihood Estimation (MLE)  
$$D_{KL}(p_{data} \parallel  p_{\phi}) := \int p_{data}(\mathbf{x}) \ \mathrm{log}\frac{p_{data}(\mathbf{x})}{p_{\phi}(\mathbf{x})} \mathrm{d}\mathbf{x}$$

**Reverse Kullback-Leibler (KL) divergence**  
Mode seeking  
$$D_{KL}(p_{\phi} \parallel p_{data}) := \int p_{\phi}(\mathbf{x}) \ \mathrm{log}\frac{p_{\phi}(\mathbf{x})}{p_{data}(\mathbf{x})} \mathrm{d}\mathbf{x}$$

**Jensen–Shannon (JS) Divergence**  
Smooth and symmetric measure, avoids the unbounded penalties of KL  
$$D_{JS}(p_{data} \parallel  p_{\phi}) := \frac{1}{2} D_{KL}\left (p_{data} \parallel \frac{1}{2}(p_{data} + p_{\phi}) \right ) + \frac{1}{2} D_{KL} \left (p_{\phi} \parallel \frac{1}{2}(p_{data} + p_{\phi}) \right )$$

**Fisher Divergence**  
Measures the discrepancy between the data and model score functions  
$$D_{F}(p_{data} \parallel p_{\phi}) := \int p_{data}(\mathbf{x}) \ \left\| \nabla_{\mathbf{x}} \mathrm{log} p_{data}(\mathbf{x}) - \nabla_{\mathbf{x}} \mathrm{log} p_{\phi}(\mathbf{x}) \right\|_{2}^{2} \mathrm{d}\mathbf{x}$$

**Total Variation (TV) Divergence**  
Captures the largest possible probability difference between the data and model  
$$D_{TV}(p_{data} \parallel p_{\phi}) := \frac{1}{2} \int_{\mathbb{R}^D} \| p_{data} - p_{\phi} \| \mathrm{d}\mathbf{x} = \underset{A \subset \mathbb{R}^D}{\mathrm{sup}} \| p_{data}(A) - p_{\phi}(A) \| $$

**Earth-Mover (EM) distance (= Wasserstein-1 distance)**  
Measures the minimal cost of moving probability mass from data to model, depend on the geometry of the sample space and remain meaningful even when the supports of data and model do not overlap  
$$W(p_{data}, p_{\phi}) := \underset{\gamma \in \prod(p_{data}, p_{\phi})}{\mathrm{inf}}  \mathbb{E}_{(x, y) \sim \gamma} \left [ \left\| x - y \right\| \right ] $$


### 1.1.2. Challenges in Modeling Distributions

For $$p_{\phi}$$ to be a valid probability density function, it must satisfy 2 fundamental properties: *Non-Negativity*, *Normalization*.  

Non-Negativity: $$p_{\phi}(\mathbf{x}) \geq 0$$ for all $$\mathbf{x}$$ in the domain.  
Normalization: $$\int p_{\phi}(\mathbf{x}) \mathrm{d}\mathbf{x} = 1$$.  

Ensuring Non-Negativity  
Apply a positive function (ex. exponential function) to the raw output of the neural network $$E_{\phi}(\mathbf{x})$$.  
$$\tilde{p}_{\phi}(\mathbf{x}) = \mathrm{exp}(E_{\phi} (\mathbf{x}))$$

Enforcing Normalization  
Divide by normalizing constant or partition function $$Z(\phi) := \int \mathrm{exp}(E_{\phi} (\mathbf{x}')) \mathrm{d}\mathbf{x}'$$.  
$$p_{\phi}(\mathbf{x}) = \frac{\tilde{p}_{\phi}(\mathbf{x})}{\int \tilde{p}_{\phi}(\mathbf{x'}) \mathrm{d}\mathbf{x}'} = \frac{\mathrm{exp}(E_{\phi} (\mathbf{x}))}{\int \mathrm{exp}(E_{\phi} (\mathbf{x}')) \mathrm{d}\mathbf{x}'} = \frac{\mathrm{exp}(E_{\phi} (\mathbf{x}))}{Z(\phi)}$$


## 1.2. Prominent Deep Generative Models
<img width="90%" alt="Chapter 1 DGMs Overview" src="https://github.com/user-attachments/assets/e24c556b-3047-4427-bf33-6e603e7cd3be">

Major challenge: computing the normalizing constant $$Z(\phi)$$ is intractable.  
→ circumvent or reduce the computational cost of evaluating the normalizing constant.  

**Energy-Based Models (EBMs)**  
Define a probability distribution through an energy function $$E_{\phi}(\mathbf{x})$$ that assigns lower energy to more probable data points.  
$$p_{\phi}(\mathbf{x}) := \frac{1}{Z(\phi)}\mathrm{exp}(-E_{\phi} (\mathbf{x})), \quad Z(\phi) = \int \mathrm{exp}(-E_{\phi} (\mathbf{x})) \mathrm{d}\mathbf{x}$$

Cons: intractablity of the partition function  

**Autoregressive Models (ARs)**  
Factorize the model distribution into a product of conditional probabilities using the chain rule of probability.  
$$p_{\phi}(\mathbf{x}) = \prod_{i=1}^{D} p_{\phi} (x_i | \mathbf{x}_{< i})$$

Pros: exact likelihood computation  
(each term $$p_{\phi} (x_i | \mathbf{x}_{< i})$$ is normalized by design)  
Cons: low sampling speed due to sequential nature, restrict flexibility due to fixed ordering  

**Variational Autoencoders (VAEs)**  
Maximize a tractable surrogate to the true log-likelihood, the Evidence Lower Bound (ELBO).  
$$L_{ELBO}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_{\theta}(\mathbf{z} | \mathbf{x})}[\mathrm{log} \ p_{\phi} (\mathbf{x} | \mathbf{z})] - D_{KL}(q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p_{prior}(\mathbf{z}))$$

Pros: combine neural networks with latent variable models  
Cons: limited sample sharpness, training pathologies  
(posterior collapse: the tendency of the encoder to ignore latent variables)  

**Normalizing Flows (NFs)**  
Learn a bijective mapping $$f_{\phi}$$ between a simple latent distribution $$\mathbf{z}$$ and a complex data distribution $$\mathbf{x}$$ via an invertible operator.  
Leverage the change-of-variable formula for densities, enabling MLE training.  
$$\mathrm{log} \ p_{\phi} (\mathbf{x}) = \mathrm{log} \ p(\mathbf{z}) + \mathrm{log} \left | \mathrm{det} \frac{\partial f_{\phi}^{-1} (\mathbf{x})}{\partial \mathbf{x}} \right |, \quad \mathbf{x} = f_{\phi} (\mathbf{z})$$

Pros: exact and tractable likelihood computation  
Cons: challenges when scaling to high-dimensional data  

**Generative Adversarial Networks (GANs)**  
Consists of 2 neural networks, a generator $$G_{\phi}$$ and a discriminator $$D_{\zeta}$$.  
Generator aims to create realistic samples $$G_{\phi}(\mathbf{z})$$ from random noise $$\mathbf{z} \sim p_{prior}$$.  
Discriminator aims to distinguish between real samples $$\mathbf{x}$$ and generated samples $$G_{\phi}(\mathbf{z})$$.  
$$\underset{G_{\phi}}{\mathrm{min}} \ \underset{D_{\zeta}}{\mathrm{max}} \ \mathbb{E}_{\mathrm{x} \sim p_{data}(\mathrm{x})} \left [ \mathrm{log} \ D_{\zeta} (\mathbf{x}) \right ] + \mathbb{E}_{\mathrm{z} \sim p_{prior}(\mathrm{z})} \left [ \mathrm{log} \left (1 - \ D_{\zeta} (G_{\phi}(\mathbf{z})) \right ) \right ]$$

Pros: do not define an explicit density function  
(from a divergence perspective, adversarial training minimizes a family of f-divergences, placing GANs within the same divergence-minimization framework)  
Cons: unstable training process due to min-max training  


## 1.3. Taxonomy of Modelings
<img width="90%" alt="Table 1.1" src="https://github.com/user-attachments/assets/e3df99ae-1892-4381-b17a-041956b9e29d">

**Explicit Models**  
Directly parameterize a probability distribution $$p_{\phi}(\mathbf{x})$$ via a tractable or approximately tractable density or mass function.  

**Implicit Models**  
Specify a distribution only through a sampling procedure.  
Thus, $$p_{\phi}(\mathbf{x})$$ is not available in closed form and may not be defined at all.



# 2. Variational Perspective: From VAEs to DDPMs

## 2.1. Variational Autoencoder
<img width="90%" alt="VAE" src="https://github.com/user-attachments/assets/d0ec8681-4142-465b-9df4-6842c8baf461">

Assumes that each observation $$\mathbf{x}$$ is generated from a latent variable sampled from a simple prior distribution, typically a standard Gaussian, $$\mathbf{z} \sim p_{prior} := \mathcal{N}(\mathbf{0}, \mathbf{I})$$.  
The decoder is kept simple, so that encoder can learn to extract useful latent features.  

### 2.1.1. Evidence Lower Bound (ELBO)

**Training via the ELBO**  
$$\mathrm{log} \ p_{\phi}(\mathbf{x}) = \mathrm{log} \int p_{\phi}(\mathbf{x}, \mathbf{z}) \mathrm{d}\mathbf{z} = \mathrm{log} \int q_{\theta}(\mathbf{z} | \mathbf{x}) \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \mathrm{d}\mathbf{z}$$
$$= \mathrm{log} \ \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ] \geq \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ]$$
$$= \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x} | \mathbf{z}) \frac{p_{\phi}(\mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ] = \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x} | \mathbf{z}) \right ] - D_{KL}( q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p(\mathbf{z}))$$

Reconstruction term ($$\mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x} | \mathbf{z}) \right ]$$): encourages accurate recovery of $$\mathbf{x}$$ from its latent code $$\mathbf{z}$$.  
Regularization term ($$D_{KL}( q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p_{\phi}(\mathbf{z}))$$): encourages the encoder distribution $$q_{\theta}(\mathbf{z} | \mathbf{x})$$ to stay close to a simple Gaussian prior $$p(\mathbf{z})$$.  


**Information-Theoretic View: ELBO as a Divergence Bound**  
$$\mathrm{log} \ p_{\phi}(\mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x}) \right ] = \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{p_{\phi}(\mathbf{z} | \mathbf{x})} \frac{q_{\theta}(\mathbf{z} | \mathbf{x})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ] $$
$$= \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \frac{q_{\theta}(\mathbf{z} | \mathbf{x})}{p_{\phi}(\mathbf{z} | \mathbf{x})} \right ] = \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ] + D_{KL}(q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p_{\phi}(\mathbf{z} | \mathbf{x}))$$
$$\geq \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ \frac{p_{\phi}(\mathbf{x}, \mathbf{z})}{q_{\theta}(\mathbf{z} | \mathbf{x})} \right ] = \mathbb{E}_{\mathbf{z} \sim q_{\theta}(\mathbf{z} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x} | \mathbf{z}) \right ] - D_{KL}( q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p(\mathbf{z}))$$
$$\mathrm{log} \ p_{\phi}(\mathbf{x}) = L_{ELBO}(\theta, \phi; \mathbf{x}) + D_{KL}(q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p_{\phi}(\mathbf{z} | \mathbf{x}))$$
($$D_{KL}(q_{\theta}(\mathbf{z} | \mathbf{x}) \parallel p_{\phi}(\mathbf{z} | \mathbf{x}))$$: inference error)  

If we fix $$\phi$$ and only update $$\theta$$, then maximizing the ELBO corresponds to maximizing the likelihood by reducing inference error.  
Since we jointly update $$\phi, \theta$$, maximizing the ELBO does not correspond to maximizing the likelihood.  


### 2.1.2. Gaussian VAE

Encoder: $$q_{\theta}(\mathbf{z} | \mathbf{x}) := \mathcal{N}(\mathbf{z}; \mu_{\theta}(\mathbf{x}), \mathrm{diag}(\sigma_{\theta}^2(\mathbf{x})))$$  
Decoder: $$p_{\phi}(\mathbf{x} | \mathbf{z}) := \mathcal{N}(\mathbf{x}; \mu_{\phi}(\mathbf{z}), \sigma^2 \mathbf{I})$$  


### 2.1.3. Drawbacks of Standard VAE

**Blurry Generations in VAEs**  
To understand this phenomenon, consider a fixed Gaussian encoder $$q_{enc}(\mathbf{z} | \mathbf{x})$$ and a decoder $$p_{dec}(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{x}; \mu(\mathbf{z}), \sigma^2 \mathbf{I})$$.  
Then optimizing the ELBO reduces to minimizing the expected reconstruction error: $$\underset{\mu}{\mathrm{argmin}} \ \mathbb{E}_{p_{data}(\mathbf{x}) q_{enc}(\mathbf{z} | \mathbf{x})} \left [  \left\| \mathbf{x} - \mu(\mathbf{z}) \right\|^2 \right ]$$.  

$$\underset{\mu}{\mathrm{argmin}} \ \mathbb{E}_{p_{data}(\mathbf{x}) q_{enc}(\mathbf{z} | \mathbf{x})} \left [  \left\| \mathbf{x} - \mu(\mathbf{z}) \right\|^2 \right ] = \underset{\mu}{\mathrm{argmin}} \ \mathbb{E}_{q_{enc}(\mathbf{x}, \mathbf{z})} \left [  \left\| \mathbf{x} - \mu(\mathbf{z}) \right\|^2 \right ]$$
$$= \underset{\mu}{\mathrm{argmin}} \ \mathbb{E}_{q_{enc}(\mathbf{z}) q_{enc}(\mathbf{x} | \mathbf{z}) } \left [  \left\| \mathbf{x} - \mu(\mathbf{z}) \right\|^2 \right ]$$

$$\mu^{*}(\mathbf{z}) = \mathbb{E}_{q_{enc}(\mathbf{x} | \mathbf{z})}[\mathbf{x}] = \int q_{enc}(\mathbf{x} | \mathbf{z}) \ \mathbf{x} \ \mathrm{d}\mathbf{x} = \int \frac{q_{enc}(\mathbf{z} | \mathbf{x}) p_{data}(\mathrm{x})}{p_{prior}(\mathrm{z})} \ \mathbf{x} \ \mathrm{d}\mathbf{x}$$
$$= \frac{1}{p_{prior}(\mathrm{z})} \int q_{enc}(\mathbf{z} | \mathbf{x}) \ p_{data}(\mathrm{x}) \ \mathbf{x} \ \mathrm{d}\mathbf{x} = \frac{ \mathbb{E}_{p_{data} (\mathrm{x})} \left [ q_{enc}(\mathbf{z} | \mathbf{x}) \ \mathbf{x} \right ] }{ \mathbb{E}_{p_{data} (\mathrm{x})} \left [ q_{enc}(\mathbf{z} | \mathbf{x}) \right ]}$$

Suppose that two distinct inputs $$\mathbf{x}, \mathbf{x}'$$ are mapped to overlapping regions in latent space.  
(the supports of $$q_{enc}(\cdot | \mathbf{x})$$ and $$q_{enc}(\cdot | \mathbf{x}')$$ intersect)  
→ $$\mu^{*}(\mathbf{z})$$ averages over multiple inputs, which leads to blurry outputs.  


### 2.1.4. From Standard VAE to Hierarchical VAEs
<img width="90%" alt="HVAE" src="https://github.com/user-attachments/assets/31d2b5c4-b8d2-4635-8b22-01c4854bdbdf">

Unlike VAEs that use a single latent code $$\mathbf{z}$$, hierarchical VAEs (HVAEs) introduce multiple layers of latent variables arranged in a top-down hierarchy.  

$$p_{\mathrm{HVAE}}(\mathbf{x}) := \int p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L}) \mathrm{d}\mathbf{z}_{1:L}$$
$$p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L}) = p_{\phi}(\mathbf{x} | \mathbf{z}_1) \prod_{i=2}^{L} p_{\phi}(\mathbf{z}_{i-1} | \mathbf{z}_{i}) \ p(\mathbf{z}_L)$$
$$q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x}) = q_{\theta}(\mathbf{z}_{1} | \mathbf{x}) \prod_{i=2}^{L} q_{\theta}(\mathbf{z}_{i} | \mathbf{z}_{i-1}) $$


**HVAE's ELBO**  
$$\mathrm{log} \ p_{\mathrm{HVAE}}(\mathbf{x}) = \mathrm{log} \int p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L}) \mathrm{d}\mathbf{z}_{1:L}$$
$$= \mathrm{log} \int \frac{p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x}) \mathrm{d}\mathbf{z}_{1:L} = \mathrm{log} \ \mathbb{E}_{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \left [ \frac{p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \right ]$$
$$\geq \mathbb{E}_{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \left [ \mathrm{log} \frac{p_{\phi}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \right ] = \mathbb{E}_{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \left [ \mathrm{log} \frac{p_{\phi}(\mathbf{x} | \mathbf{z}_1) \prod_{i=2}^{L} p_{\phi}(\mathbf{z}_{i-1} | \mathbf{z}_{i}) \ p(\mathbf{z}_L)}{q_{\theta}(\mathbf{z}_{1} | \mathbf{x}) \prod_{i=2}^{L} q_{\theta}(\mathbf{z}_{i} | \mathbf{z}_{i-1})} \right ]$$

$$= \mathbb{E}_{q_{\theta}(\mathbf{z}_{1:L} | \mathbf{x})} \left [ \mathrm{log} \ p_{\phi}(\mathbf{x} | \mathbf{z}_1) + \mathrm{log} \frac{ p_{\phi}(\mathbf{z}_{1} | \mathbf{z}_{2}) }{q_{\theta}(\mathbf{z}_{1} | \mathbf{x}) } + \sum_{i=2}^{L-1} \mathrm{log} \frac{ p_{\phi}(\mathbf{z}_{i} | \mathbf{z}_{i+1})}{q_{\theta}(\mathbf{z}_{i} | \mathbf{z}_{i-1})} + \mathrm{log} \frac{p(\mathbf{z}_L)}{q_{\theta}(\mathbf{z}_{L} | \mathbf{z}_{L-1})} \right ]$$


**Why Deeper Networks in a Flat VAE are Not Enough**  
2 fundamental limitations of a standard flat VAE that are not resolved by simply making the encoder and decoder deeper.  

Limitation 1: variational family  
Encoder posterior is a single Gaussian with diagonal covariance.  
Since increasing the layer depth does not expand the family, they cannot match multi-peaked $$p_{\theta}(\mathbf{z} | \mathbf{x})$$.  

Limitation 2: posterior collapse  
Maximizing ELBO can make the decoder to model data well without using $$\mathbf{z}$$.  
Increasing the layers of decoder increases the expressiveness, which makes "ignoring $$\mathbf{z}$$" solution more serious.  


**What Hierarchy Changes?**  
Each inference conditional $$q_{\theta}(\mathbf{z}_{i} | \mathbf{z}_{i-1})$$ is aligned with its top-down generative countepart $$p_{\phi}(\mathbf{z}_{i} | \mathbf{z}_{i+1})$$.  
This distributes the information penalty across levels and localizes learning signals through these adjacent KL terms.  
These properties stem from the hierarchical latent graph, not from simply deepening networks in flat VAE.  


**2 Limitations of HVAEs**  

Limitation 1: training difficulty  
Since encoder and decoder must be optimized jointly, lower layers and the decoder can already reconstruct $$\mathbf{x}$$, leaving higher-level latetns with little effective signal.  

Limitation 2: model design  
Overly expressive conditionals can dominate the reconstruction task and suppress the utility of higher latents.  



---

The integral over $$\mathbf{z}$$ is intractable: no closed-form solution (analytic integration impossible), numerical integration computationally infeasible.  

1. Variational View (VAE, DPM, DDPM)
  frames diffusion as learning a denoising process through a variational objective

2. Score-Based View (EBM, NCSN, Score SDE)
  learns the score function, the gradient of the log density, which guides how to gradually remove noise from samples
  connects diffusion modeling with classical differential equation theory

3. Flow-Based View (NF, NODE, FM)
  generation as a continuous transformation that transports samples from a simple prior toward the data distribution, where evolution is governed by a velocity field through an ODE, which explicitly defines how probability mass moves over time
  extends beyond prior-to-data generation to more general distribution-to-distribution translation problems

diffusion models can be viewed as approaches for transporting one distribution to another
→ connections to classical optimal transport and the Schrödinger bridge, interpreted as optimal transport with entropy regularization