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

- Trajectory: function that maps time to some location in space.  
  $$X: [0, 1] \rightarrow \mathbb{R}^d, t \mapsto X_t$$  

- Vector field: function that maps time and location to a velocity in space.  
  $$u: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, (x, t) \mapsto u_t(x)$$  

- Ordinary Differential Equation (ODE): imposes a condition on a trajectory $$X$$ that "follows along the lines" of the vector field $$u_t$$, starting at the point $$x_0$$.  
  ODE: $$\frac{d}{dt} X_t = u_t(X_t)$$  
  Initial conditions: $$X_0 = x_{0}$$  

- Flow: collections of solutions to an ODE for lots of initial conditions.  
  $$\psi: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, (x_0, t) \mapsto \psi_t(x_0)$$  
  Flow ODE: $$\frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0))$$  
  Flow initial conditions: $$\psi_0(x_0) = x_0$$  


Does a solution exist and if so, is it unique?  
In machine learning, unique solutions to ODEs/flows exist.  
(Picard–Lindelöf Theorem)  

Since it is not possible to compute the flow explicitly, use numerical methods to simulate ODEs.  
(ex. Euler method: $$X_{t+h} = X_t + h u_t(X_t)$$)

- Flow model: function that maps a simple distribution into the data distribution by simulating an ODE.  
  $$p_{init} \xrightarrow[ODE]{} p_{data}$$  
  Neural network: $$u^{\theta}_t: \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d$$  
  Random initialization: $$X_0 \sim p_{init}$$  
  ODE: $$\frac{d}{dt} X_t = u_t^{\theta}(X_t)$$  
  Goal: make the endpoint $$X_1$$ of the trajectory have distribution $$p_{data}$$  
  $$X_1 \sim p_{data} \Leftrightarrow \psi_t^{\theta}(X_0) \sim p_{data}$$  


<details><summary>Sampling from a Flow Model with Euler Method</summary>
<img width="70%" alt="Algorithm1" src="https://github.com/user-attachments/assets/067e8b46-e66d-4a31-a160-5a4efc0cb99d">
</details>




# Proof

- Picard–Lindelöf Theorem  
  