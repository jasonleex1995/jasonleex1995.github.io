---
title: "Introduction to Flow Matching and Diffusion Models"
parent: Videos
layout: default
---

# Table of Contents

> Links: [project page](https://diffusion.csail.mit.edu/), [YouTube playlist](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH)  

- [1. From Generation to Sampling](#1-from-generation-to-sampling)  
- [2. Flow and Diffusion Models](#2-flow-and-diffusion-models)  
  - [2.1. Flow Models](#21-flow-models)  
  - [2.2. Diffusion Models](#22-diffusion-models)  
- [3. Constructing a Training Target](#3-constructing-a-training-target)  
  - [3.1. Conditional and Marginal Probability Path](#31-conditional-and-marginal-probability-path)
  - [3.2. Conditional and Marginal Vector Fields](#32-conditional-and-marginal-vector-fields)
  - [3.3. Conditional and Marginal Score Functions](#33-conditional-and-marginal-score-functions)
- [4. Training Flow and Diffusion Models](#4-training-flow-and-diffusion-models)  
  - [4.1. Flow Matching](#41-flow-matching)  
  - [4.2. Score Matching](#42-score-matching)  


---
# 1. From Generation to Sampling  

What is generative modeling?  
Model the generation as sampling from the data distribution.  
Simply, $$z \sim p_{\mathrm{data}}$$.  

How to generate samples?  
Transform samples from an initial distribution into samples from the data distribution.  
Simply, generate by converting $$x \sim p_{\mathrm{init}}$$ into $$z \sim p_{\mathrm{data}}$$.  


---
# 2. Flow and Diffusion Models

Simulating a differential equation can transform an initial distribution into the data distribution.  
Simulating ordinary differential equations (ODEs) → flow matching  
Simulating stochastic differential equations (SDEs) → diffusion models  


## 2.1. Flow Models
### 2.1.1. About Flow

- Trajectory  
  Function that maps time to some location in space.  
  $$X: [0, 1] \rightarrow \mathbb{R}^d, \ t \mapsto X_t$$  

- Vector Field  
  Function that maps time and location to a velocity in space.  
  $$u: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \ (x, t) \mapsto u_t(x)$$  

- Ordinary Differential Equation (ODE)  
  Imposes a condition on a trajectory $$X$$ that "follows along the lines" of the vector field $$u_t$$, starting at the point $$x_0$$.  
  $$X_0 = x_{0}, \ \frac{d}{dt} X_t = u_t(X_t)$$  

- Flow  
  Collections of solutions to an ODE for lots of initial conditions.  
  $$\psi: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \ (x_0, t) \mapsto \psi_t(x_0)$$  
  $$\psi_0(x_0) = x_0, \ \frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0))$$  


### 2.1.2. ODE Solution Existence and Uniqueness

Does a solution exist and if so, is it unique?  
In machine learning, unique solutions to ODEs exist.  
(see more details on proof)  


### 2.1.3. Simulating ODE

Euler method: $$X_{t+h} = X_t + h u_t(X_t)$$


### 2.1.4. Flow Models

Initialization: $$X_0 \sim p_{\mathrm{init}}$$  
Simulation: $$\frac{d}{dt} X_t = u_t^{\theta}(X_t)$$  
Goal: $$X_1 \sim p_{\mathrm{data}}$$  


<details><summary>Sampling from a Flow Model (Euler method)</summary>
<img width="70%" alt="Algorithm1" src="https://github.com/user-attachments/assets/067e8b46-e66d-4a31-a160-5a4efc0cb99d">
</details>


## 2.2. Diffusion Models
### 2.2.1. About SDE

- Stochastic Trajectory (= stochastic process, random trajectory)  
  $$X_t$$ is a random variable for every $$0 \leq t \leq 1$$.  
  $$X: [0, 1] \rightarrow \mathbb{R}^d, \ t \mapsto X_t$$  

- Brownian Motion (= Wiener process)  
  Continuous stochastic process, but not differentiable due to stochasticity.  
  $$W = (W_t)_{t \geq 0}$$ is characterized by following 3 properties.  
  1. Initial condition: $$W_0 = 0$$  
  2. Gaussian increments: $$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$$ for all $$0 \leq s < t$$.  
  3. Independent increments: $$W_{t_1} - W_{t_0}, \cdots W_{t_n} - W_{t_{n-1}}$$ are independent for any $$0 \leq t_0 < t_1 < \cdots < t_n$$.  

- Diffusion Coefficient  
  Inject stochasticity (randomness) into ODE.  
  $$\sigma: [0, 1] \rightarrow \mathbb{R}_{\geq 0}, \ t \mapsto \sigma_t$$    

- Stochastic Differential Equation (SDE)  
  Extend the deterministic dynamics of an ODE by adding stochastic dynamics driven by Brownian motion.  
  $$X_0 = x_{0}, \ dX_t = u_t(X_t)dt + \sigma_t dW_t$$  


### 2.2.2. SDE Solution Existence and Uniqueness

Does a solution exist and if so, is it unique?  
In machine learning, unique solutions to SDEs exist.  
(see more details on proof)  


### 2.2.3. Simulating SDE

Euler-Maruyama method: $$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma_t \epsilon_t, \ \epsilon_t \sim \mathcal{N}(0, I_d)$$  


### 2.2.4. Diffusion Models

Initialization: $$X_0 \sim p_{\mathrm{init}}$$  
Simulation: $$dX_t = u_t^{\theta}(X_t) dt + \sigma_t dW_t$$  
Goal: $$X_1 \sim p_{\mathrm{data}}$$  


<details><summary>Sampling from a Diffusion Model (Euler-Maruyama method)</summary>
<img width="75%" alt="Algorithm2" src="https://github.com/user-attachments/assets/57645d75-dd31-4bf4-b746-13fe82af1690">
</details>


---
# 3. Constructing a Training Target

Goal: find an equation for the training target $$u_t^{\mathrm{target}}$$ such that the corresponding ODE/SDE converts $$p_{\mathrm{init}}$$ into $$p_{\mathrm{data}}$$  

Key terminology  
Conditional = per single data point  
Marginal = across distribution of data points  

<details><summary>Summary - Conditional Probability Path, Vector Field, Score</summary>
<img width="100%" alt="Conditional Probability Path" src="https://github.com/user-attachments/assets/b6e41cc2-075a-453c-b14c-7d81ef64fd52">
</details>

<details><summary>Summary - Marginal Probability Path, Vector Field, Score</summary>
<img width="100%" alt="Marginal Probability Path" src="https://github.com/user-attachments/assets/9b5c693b-a71c-4fe5-99b5-df683c3961b0">
</details>


## 3.1. Conditional and Marginal Probability Path
<img width="100%" alt="Probability Path" src="https://github.com/user-attachments/assets/0ae82300-f04f-4123-beed-25ad66412859">

- Probability path  
  Path from initial distribution $$p_{\mathrm{init}}$$ to data distribution $$p_{\mathrm{data}}$$.  

- Conditional probability path: $$p_{t}(\cdot | z)$$   
  Given a single data point $$z$$, path from initial distribution $$p_{\mathrm{init}}$$ to a single data point distribution $$\delta_z$$.  
  
- Marginal probability path: $$p_t$$  
  Interpolates between $$p_{\mathrm{init}}$$ and $$p_{\mathrm{data}}$$.  
  Sampling from marginal path: $$z \sim p_{\mathrm{data}}, \ x \sim p_t(\cdot | z) \ \Rightarrow \ x \sim p_t$$  
  Density of marginal path: $$p_t(x) = \int p_t(x|z) \ p_{\mathrm{data}}(z) \ dz$$  
  Noise-data interpolation: $$p_0 = p_{\mathrm{init}}, \ p_1 = p_{\mathrm{data}}$$  
  

## 3.2. Conditional and Marginal Vector Fields
<img width="100%" alt="ODE trajectory" src="https://github.com/user-attachments/assets/00bc7de6-63e9-46c8-ab75-ba5e676e4f8a">

For every data point $$z \in \mathbb{R}^d$$,
let $$u_t^{\mathrm{target}}(\cdot|z)$$ denote an ***conditional vector field***,
defined so that the corresponding ODE yields the conditional probability path $$p_t(\cdot | z)$$.  
Simply, $$X_0 \sim p_{\mathrm{init}}, \ \frac{d}{dt}X_t = u_t^{\mathrm{target}}(X_t|z) \ \Rightarrow \ X_t \sim p_t(\cdot | z) \ (0 \leq t \leq 1)$$  

If we set the ***marginal vector field*** as $$u_t^{\mathrm{target}}(x) = \int u_t^{\mathrm{target}} (x|z) \ \frac{p_t(x|z) \ p_{\mathrm{data}}(z)}{p_t(x)} \ dz$$, 
then the ODE follows the marginal probability path $$p_t$$.  
Simply, $$X_0 \sim p_{\mathrm{init}}, \ \frac{d}{dt} X_t = u_t^{\mathrm{target}}(x) \ \Rightarrow \ X_t \sim p_t \ (0 \leq t \leq 1)$$.  
In particular, $$X_1 \sim p_{\mathrm{data}}$$ for this ODE, so we can say "$$u_t^{\mathrm{target}}(x)$$ converts $$p_{\mathrm{init}}$$ into $$p_{\mathrm{data}}$$".  

We can prove this by using continuity equation.  
(see more details on proof)  


## 3.3. Conditional and Marginal Score Functions

Conditional score function: 
$$\nabla_x \mathrm{log} p_t(x | z)$$  
Marginal score function: 
$$\nabla_x \mathrm{log} p_t(x) = \int \nabla_x \mathrm{log} p_t(x|z) \frac{\ p_t(x|z) \ p_{\mathrm{data}}(z)}{p_t(x)} \ dz$$  

Define the conditional and marginal vector fields $$u_t^{\mathrm{target}} (x|z)$$ and $$u_t^{\mathrm{target}} (x)$$ as before. Then, for diffusion coefficient $$\sigma_t \geq 0$$, we may construct an SDE which follows the same probability path.  
Simply, $$X_0 \sim p_{\mathrm{init}}, \ dX_t = [u_t^{\mathrm{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \mathrm{log} p_t(X_t) ]dt + \sigma_t dW_t \ \Rightarrow \ X_t \sim p_t \ (0 \leq t \leq 1)$$.  
In particular, $$X_1 \sim p_{\mathrm{data}}$$ for this SDE.  

We can prove this by using Fokker-Planck equation.  
(see more details on proof)  


When the probability path is static ($$p_t = p$$), 
we set $$u_t^{\mathrm{target}}(X_t) = 0$$ 
and obtain the SDE $$dX_t = \frac{\sigma_t^2}{2} \nabla \mathrm{log} p_t(X_t) dt + \sigma_t dW_t$$,
which is commonly known as ***Langevin dynamics***.  


---
# 4. Training Flow and Diffusion Models
## 4.1. Flow Matching

ODE: $$X_0 \sim p_{\mathrm{init}}, \ dX_t = u_t^{\theta}(X_t) dt$$  
Flow matching loss: $$L_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, x \sim p_t} [|| u_t^{\theta}(x) - u_t^{\mathrm{target}}(x) ||^2]$$  
Conditional flow matching loss: $$L_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot | z)} [|| u_t^{\theta}(x) - u_t^{\mathrm{target}}(x|z) ||^2]$$  

$$L_{\mathrm{FM}}(\theta) = L_{\mathrm{CFM}}(\theta) + C$$
, where $$C$$ is independent of $$\theta$$.  
$$\nabla_{\theta} L_{\mathrm{FM}}(\theta) = \nabla_{\theta} L_{\mathrm{CFM}}(\theta)$$  

Minimizing $$L_{\mathrm{CFM}}(\theta)$$ is equivalent to minimizing $$L_{\mathrm{FM}}(\theta)$$  
By explicitly regressing against the tractable conditional vector field, we are implicitly regressing against the intractable marginal vector field.  

<details><summary>Flow Matching Training Algorithm (Gaussian Conditional Optimal Transport path)</summary>
<img width="100%" alt="flow matching" src="https://github.com/user-attachments/assets/6bd6b4db-b089-4249-b169-30388fa8788b">
</details>


## 4.2. Score Matching

SDE: $$X_0 \sim p_{\mathrm{init}}, \ dX_t = [u_t^{\theta}(X_t) + \frac{\sigma_t^2}{2} s_t^{\theta}(X_t)] dt + \sigma_t dW_t$$  
Score matching loss: $$L_{\mathrm{SM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, x \sim p_t} [|| s_t^{\theta}(x) - \nabla \mathrm{log} p_t(x) ||^2]$$  
Conditional score matching loss (= denoising score matching loss): $$L_{\mathrm{CSM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot | z)} [|| s_t^{\theta}(x) - \nabla \mathrm{log} p_t(x|z) ||^2]$$  

$$L_{\mathrm{SM}}(\theta) = L_{\mathrm{CSM}}(\theta) + C$$
, where $$C$$ is independent of $$\theta$$.  
$$\nabla_{\theta} L_{\mathrm{SM}}(\theta) = \nabla_{\theta} L_{\mathrm{CSM}}(\theta)$$  

For Gaussian probability path $$p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$$, 
it holds that the vector field can be converted into the score.  
$$u_t^{\mathrm{target}}(x|z) = (\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \nabla \mathrm{log} p_t(x|z) + \frac{\dot{\alpha}_t}{\alpha_t}x$$  
$$u_t^{\mathrm{target}}(x) = (\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \nabla \mathrm{log} p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t}x$$  

For Gaussian probability paths, there is no need to separately train both the marginal score and the marginal vector field, as knowledge of one is sufficient to compute the other.  

<details><summary>Score Matching Training Algorithm (Gaussian path)</summary>
<img width="100%" alt="score matching" src="https://github.com/user-attachments/assets/14350379-f4f6-4242-9a39-cfe187c81285">
</details>


---
# Proof
## Flow and Diffusion Models

- Existence and Uniqueness Theorem of ODEs  
  Proof: Picard–Lindelöf Theorem  
  If the vector field is Lipschitz, then a unique solution to the ODE exists.  

- From ODEs to SDEs  
  $$\frac{d}{dt} X_t = u_t(X_t)$$  
  $$\frac{1}{h} (X_{t+h} - X_t) = u_t(X_t) + R_t(h)$$  
  $$X_{t+h} = X_t + h u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h R_t(h)$$  
  $$dX_t = u_t(X_t)dt + \sigma_t dW_t$$  

- Existence and Uniqueness Theorem of SDEs  
  Proof: Construct solutions via stochastic integrals and Ito-Riemann sums.  
  If the vector field and diffusion coefficient are Lipschitz, then a unique solution to the SDE exists.  


## Constructing a Training Target

- Continuity Equation  
  <img width="100%" alt="Continuity Equation" src="https://github.com/user-attachments/assets/e0f3201e-a520-4a15-a571-da36dd304edf">  
  Use the continuity equation of conditional vector field.  
  $$\partial_t p_t(x) = \partial_t \int p_t(x|z) \ p_{\mathrm{data}}(z) dz$$  
  $$ = \int \partial_t p_t(x|z) \ p_{\mathrm{data}}(z) dz$$  
  $$ = \int - \mathrm{div} \left ( p_t(\cdot | z) \ u_t^{\mathrm{target}} (\cdot | z) \right ) (x) \ p_{\mathrm{data}}(z) \ dz$$  
  $$ = - \mathrm{div} \left ( \int p_t(x | z) \ u_t^{\mathrm{target}} (x | z) \ p_{\mathrm{data}}(z) \ dz \right )$$  
  $$ = - \mathrm{div} \left ( \int \ u_t^{\mathrm{target}} (x | z) \frac{p_t(x | z) \ p_{\mathrm{data}}(z)}{p_{t}(x)} \ p_t(x) \ dz \right )$$  
  $$ = - \mathrm{div} \left ( p_t u_t^{\mathrm{target}} \right ) (x)$$  


- Fokker-Planck Equation  
  <img width="100%" alt="Fokker-Planck equation" src="https://github.com/user-attachments/assets/ecab0df1-b83c-4bca-b404-a2ecae5dd2f6">  
  Use the continuity equation of marginal vector field.  
  $$\partial_t p_t(x) = - \mathrm{div} \left ( p_t u_t^{\mathrm{target}} \right ) (x)$$  
  $$ = - \mathrm{div} \left ( p_t u_t^{\mathrm{target}} \right ) (x) - \frac{\sigma_t^2}{2} \Delta p_t(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$  
  $$ = - \mathrm{div} \left ( p_t u_t^{\mathrm{target}} \right ) (x) - \mathrm{div}(\frac{\sigma_t^2}{2} \nabla p_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$  
  $$ = - \mathrm{div} \left ( p_t u_t^{\mathrm{target}} \right ) (x) - \mathrm{div}( \frac{\sigma_t^2}{2} p_t \nabla \mathrm{log} p_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$  
  $$ = - \mathrm{div} \left ( p_t \left [ u_t^{\mathrm{target}} + \frac{\sigma_t^2}{2} \nabla \mathrm{log} p_t \right ] \right ) (x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$  


- Score Function  
  $$\nabla_x \mathrm{log} p_t(x) = \frac{\nabla_x p_t(x)}{p_t(x)}$$  
  $$ = \frac{\nabla_x \int p_t(x|z) \ p_{\mathrm{data}}(z) \ dz}{p_t(x)}$$  
  $$ = \frac{\int \nabla_x p_t(x|z) \ p_{\mathrm{data}}(z) \ dz}{p_t(x)}$$  
  $$ = \frac{\int \nabla_x \mathrm{log} p_t(x|z) \ p_t(x|z) \ p_{\mathrm{data}}(z) \ dz}{p_t(x)}$$  
  $$ = \int \nabla_x \mathrm{log} p_t(x|z) \frac{\ p_t(x|z) \ p_{\mathrm{data}}(z)}{p_t(x)} \ dz$$  


- Gaussian Conditional Vector Field  
  $$X_t = \psi_t^{\mathrm{target}}(X_0 | z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta^2_t I_d) = p_t(\cdot | z)$$  
  $$\frac{d}{dt} \psi_t^{\mathrm{target}}(x | z) = u_t^{\mathrm{target}}(\psi_t^{\mathrm{target}} (x|z) \ | \ z)$$  
  $$\Leftrightarrow \ \dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\mathrm{target}}(\alpha_t z + \beta_t x | z)$$  
  $$\Leftrightarrow \ \dot{\alpha}_t z + \dot{\beta}_t (\frac{x - \alpha_t z}{\beta_t}) = u_t^{\mathrm{target}}(x|z)$$  
  $$\Leftrightarrow \ (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z + \frac{\dot{\beta}_t}{\beta_t} x = u_t^{\mathrm{target}}(x|z)$$  


## Training Flow and Diffusion Models

- $$L_{\mathrm{FM}}(\theta) = L_{\mathrm{CFM}}(\theta) + C$$  
  $$L_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, x \sim p_t} [|| u_t^{\theta}(x) - u_t^{\mathrm{target}}(x) ||^2]$$  
  $$= \mathbb{E}_{t, x} [|| u_t^{\theta}(x) ||^2 - 2 u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x) + || u_t^{\mathrm{target}}(x) ||^2]$$  
  $$= \mathbb{E}_{t, x} [|| u_t^{\theta}(x) ||^2]  - 2 \mathbb{E}_{t, z, x} [u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x)] + \mathbb{E}_{t, z, x} [|| u_t^{\mathrm{target}}(x) ||^2]$$  
  $$= \mathbb{E}_{t, x} [|| u_t^{\theta}(x) ||^2]  - 2 \mathbb{E}_{t, z, x} [u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x)] + C_1$$  
  $$\mathbb{E}_{t \sim \mathrm{Unif}, x \sim p_t} [u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x)] = \int_0^1 \int p_t(x) \ u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x) \ dx \ dt$$  
  $$= \int_0^1 \int p_t(x) \ u_t^{\theta}(x)^T \ [\int u_t^{\mathrm{target}}(x|z) \frac{p_t(x|z) p_{\mathrm{data}}(z)}{p_t(x)} dz] \ dx \ dt$$  
  $$= \int_0^1 \int \int \ u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x|z) \ p_t(x|z) \ p_{\mathrm{data}}(z) \ dz \ dx \ dt$$  
  $$= \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot | z)} [u_t^{\theta}(x)^T u_t^{\mathrm{target}}(x | z)] $$  

- Conversion Formula of Vector Field to Score for Gaussian Probability Path  
  $$u_t^{\mathrm{target}}(x|z) = (\dot{\alpha_t} - \frac{\dot{\beta}_t}{\beta_t}\alpha_t)z + \frac{\dot{\beta}_t}{\beta_t}x$$  
  $$= (\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \frac{\alpha_t z - x}{\beta_t^2} + \frac{\dot{\alpha}_t}{\alpha_t}x$$  
  $$= (\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \nabla \mathrm{log} p_t(x|z) + \frac{\dot{\alpha}_t}{\alpha_t}x$$  
  $$u_t^{\mathrm{target}}(x) = \int u_t^{\mathrm{target}}(x|z) \frac{p_t(x|z) p_{\mathrm{data}}(z)}{p_t(x)} dz$$  
  $$= \int [(\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \nabla \mathrm{log} p_t(x|z) + \frac{\dot{\alpha}_t}{\alpha_t}x] \frac{p_t(x|z) p_{\mathrm{data}}(z)}{p_t(x)} dz$$  
  $$= (\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t) \nabla \mathrm{log} p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t}x$$  