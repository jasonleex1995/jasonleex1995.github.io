---
title: "Introduction to Flow Matching and Diffusion Models"
parent: Videos
layout: default
---

> Links: [project page](https://diffusion.csail.mit.edu/), [YouTube playlist](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH)  



# 1: From Generation to Sampling  

> **Goal**: Formalize what it means to `generate` something.  

What is generative modeling?  
Model the generation as sampling from the data distribution.  
Simply, $$z \sim p_{data}$$.  

How to generate samples?  
Transform samples from an initial distribution into samples from the data distribution.  
Simply, generate by converting $$x \sim p_{init}$$ into $$z \sim p_{data}$$.  



# 2: Flow and Diffusion Models

> **Goal**: Understand `differential equations` and how we can build generative models with them.  

Simulating a differential equation can transform an initial distribution into the data distribution.  
Simulating ordinary differential equations (ODEs) → flow matching  
Simulating stochastic differential equations (SDEs) → diffusion models  


## 2.1. Flow Models
### 2.1.1. About Flow

- Trajectory  
  Function that maps time to some location in space.  
  $$X: [0, 1] \rightarrow \mathbb{R}^d, t \mapsto X_t$$  

- Vector Field  
  Function that maps time and location to a velocity in space.  
  $$u: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, (x, t) \mapsto u_t(x)$$  

- Ordinary Differential Equation (ODE)  
  Imposes a condition on a trajectory $$X$$ that "follows along the lines" of the vector field $$u_t$$, starting at the point $$x_0$$.  
  $$X_0 = x_{0}, \frac{d}{dt} X_t = u_t(X_t)$$  

- Flow  
  Collections of solutions to an ODE for lots of initial conditions.  
  $$\psi: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, (x_0, t) \mapsto \psi_t(x_0)$$  
  $$\psi_0(x_0) = x_0, \frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0))$$  


### 2.1.2. ODE Solution Existence and Uniqueness

Does a solution exist and if so, is it unique?  
In machine learning, unique solutions to ODEs exist.  


### 2.1.3. Simulating ODE

Euler method: $$X_{t+h} = X_t + h u_t(X_t)$$


### 2.1.4. Flow Models

Initialization: $$X_0 \sim p_{init}$$  
Simulation: $$\frac{d}{dt} X_t = u_t^{\theta}(X_t)$$  
Goal: $$X_1 \sim p_{data}$$  


<details><summary>Sampling from a Flow Model (Euler method)</summary>
<img width="70%" alt="Algorithm1" src="https://github.com/user-attachments/assets/067e8b46-e66d-4a31-a160-5a4efc0cb99d">
</details>


## 2.2. Diffusion Models
### 2.2.1. About SDE

- Stochastic Trajectory (= stochastic process, random trajectory)  
  $$X_t$$ is a random variable for every $$0 \leq t \leq 1$$.  
  $$X: [0, 1] \rightarrow \mathbb{R}^d, t \mapsto X_t$$  

- Brownian Motion (= Wiener process)  
  Continuous stochastic process, but not differentiable due to stochasticity.  
  $$W = (W_t)_{t \geq 0}$$ is characterized by following 3 properties.  
  1. Initial condition: $$W_0 = 0$$  
  2. Gaussian increments: $$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$$ for all $$0 \leq s < t$$.  
  3. Independent increments: $$W_{t_1} - W_{t_0}, \cdots W_{t_n} - W_{t_{n-1}}$$ are independent for any $$0 \leq t_0 < t_1 < \cdots < t_n$$.  

- Diffusion Coefficient  
  Inject stochasticity (randomness) into ODE.  
  $$\sigma: [0, 1] \rightarrow \mathbb{R}_{\geq 0}, t \mapsto \sigma_t$$    

- Stochastic Differential Equation (SDE)  
  Extend the deterministic dynamics of an ODE by adding stochastic dynamics driven by Brownian motion.  
  $$X_0 = x_{0}, dX_t = u_t(X_t)dt + \sigma_t dW_t$$  


### 2.2.2. SDE Solution Existence and Uniqueness

Does a solution exist and if so, is it unique?  
In machine learning, unique solutions to SDEs exist.  


### 2.2.3. Simulating SDE

Euler-Maruyama method: $$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma_t \epsilon_t, \epsilon_t \sim \mathcal{N}(0, I_d)$$  


### 2.2.4. Diffusion Models

Initialization: $$X_0 \sim p_{init}$$  
Simulation: $$dX_t = u_t^{\theta}(X_t) dt + \sigma_t dW_t$$  
Goal: $$X_1 \sim p_{data}$$  


<details><summary>Sampling from a Diffusion Model (Euler-Maruyama method)</summary>
<img width="75%" alt="Algorithm2" src="https://github.com/user-attachments/assets/57645d75-dd31-4bf4-b746-13fe82af1690">
</details>





# Proof

- Existence and Uniqueness Theorem of ODEs  
  Proof: Picard–Lindelöf Theorem  
  If the vector field is Lipschitz, then a unique solution to the ODE exists.  

- From ODEs to SDEs  
  $$\frac{d}{dt} X_t = u_t(X_t)$$  
  $$\frac{1}{h} (X_{t+h} - X_t) = u_t(X_t) + R_t(h)$$  
  ($$R_t(h)$$: error term such that $$\underset{h \to 0}{\mathrm{lim}} R_t(h) = 0$$)  
  $$X_{t+h} = X_t + h u_t(X_t) + \sigma_t(W_{t+h} - W_t) + h R_t(h)$$  
  $$dX_t = u_t(X_t)dt + \sigma_t dW_t$$  

- Existence and Uniqueness Theorem of SDEs  
  Proof: Construct solutions via stochastic integrals and Ito-Riemann sums.  
  If the vector field and diffusion coefficient are Lipschitz, then a unique solution to the SDE exists.  