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

  3. 

Part C & D. Controlling and Accelerating the Diffusion Sampling
Part C. Sampling of Diffusion Models
Part D. Learning Fast Generative Models


---

<img width="90%" alt="Part AB Overview" src="https://github.com/user-attachments/assets/a67fa537-8467-4f2e-a361-a7564cd84732">



# 1. Deep Generative Modeling

> *What I cannot create, I do not understand* by Richard P. Feynman


## 1.1. What is Deep Generative Modeling?

**What is Deep Generative Models (DGMs)?**  
DGMs are neural networks that learn a probability distribution to generate new samples that resemble the dataset.  

**How to train DGMs?**  
Fit the model parameter $$\phi$$ by minimizing a discrepancy that measures how far the model distribution $$p_{\phi}$$ is from the data distribution $$p_{data}$$.  

**How to generate samples using DGMs?**  
Run the model's sampling procedure to draw $$\mathbf{x} \sim p_{\phi}$$.  

**How to measure the quality of DGMs?**  
Quality of generated samples & summary statistics between generated samples and $$p_{data}$$.  


### 1.1.1. Mathematical Setup

> Assumption: a finite set of samples is drawn independently and identically distributed (i.i.d.) from the data distribution $$p_{data}$$  

$$\phi \in \underset{\phi}{\mathrm{argmin}} \ D(p_{data}, p_{\phi})$$

2 main families of $$D$$: *f-divergences*, *Optimal Transport (OT) distances*  

|Discrepancy|Definition|Meaning|  
|:-:|:-:|:-:|  
|Forward Kullback-Leibler (KL) divergence|$$D_{KL}(p_{data} \parallel  p_{\phi}) := \int p_{data}(\mathbf{x}) \ \mathrm{log}\frac{p_{data}(\mathbf{x})}{p_{\phi}(\mathbf{x})} \mathrm{d}\mathbf{x}$$|Mode covering, minimizing forward KL = Maximum Likelihood Estimation (MLE)|  
|Reverse Kullback-Leibler (KL) divergence|$$D_{KL}(p_{\phi} \parallel p_{data}) := \int p_{\phi}(\mathbf{x}) \ \mathrm{log}\frac{p_{\phi}(\mathbf{x})}{p_{data}(\mathbf{x})} \mathrm{d}\mathbf{x}$$|Mode seeking|  
|Jensen–Shannon (JS) Divergence|$$D_{JS}(p_{data} \parallel  p_{\phi}) := \frac{1}{2} D_{KL}\left (p_{data} \parallel \frac{1}{2}(p_{data} + p_{\phi}) \right ) + \frac{1}{2} D_{KL} \left (p_{\phi} \parallel \frac{1}{2}(p_{data} + p_{\phi}) \right ) $$|Smooth and symmetric measure, avoids the unbounded penalties of KL|  
|Fisher Divergence|$$D_{F}(p_{data} \parallel p_{\phi}) := \int p_{data}(\mathbf{x}) \ \left\| \nabla_{\mathbf{x}} \mathrm{log} p_{data}(\mathbf{x}) - \nabla_{\mathbf{x}} \mathrm{log} p_{\phi}(\mathbf{x}) \right\|_{2}^{2} \mathrm{d}\mathbf{x}$$|Measures the discrepancy between the data and model score functions|  
|Total Variation (TV) Divergence|$$D_{TV}(p_{data} \parallel p_{\phi}) := \frac{1}{2} \int_{\mathbb{R}^D} \| p_{data} - p_{\phi} \| \mathrm{d}\mathbf{x} = \underset{A \subset \mathbb{R}^D}{\mathrm{sup}} \| p_{data}(A) - p_{\phi}(A) \| $$|Captures the largest possible probability difference between the data and model|  
|Earth-Mover (EM) distance (= Wasserstein-1 distance)|$$W(p_{data}, p_{\phi}) := \underset{\gamma \in \prod(p_{data}, p_{\phi})}{\mathrm{inf}}  \mathbb{E}_{(x, y) \sim \gamma} \left [ \left\| x - y \right\| \right ] $$|Measures the minimal cost of moving probability mass from data to model, depend on the geometry of the sample space and remain meaningful even when the supports of data and model do not overlap|  


### 1.1.2. Challenges in Modeling Distributions

For $$p_{\phi}$$ to be a valid probability density function, it must satisfy 2 fundamental properties: *Non-Negativity*, *Normalization*.  

Non-Negativity: $$p_{\phi}(\mathbf{x}) \geq 0$$ for all $$\mathbf{x}$$ in the domain.  
Normalization: $$\int p_{\phi}(\mathbf{x}) \mathrm{d}\mathbf{x} = 1$$.  

Ensuring Non-Negativity: apply a positive function (ex. exponential function) to the raw output of the neural network $$E_{\phi}(\mathbf{x})$$.  
$$\tilde{p}_{\phi}(\mathbf{x}) = \mathrm{exp}(E_{\phi} (\mathbf{x}))$$

Enforcing Normalization: divide by normalizing constant or partition function $$Z(\phi) := \int \mathrm{exp}(E_{\phi} (\mathbf{x}')) \mathrm{d}\mathbf{x}'$$.  
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




