$$
\newcommand{\E}{\mathbb E}
$$



# Paper-narrative : constraint sampling with projection



**Notations :** 

- $x \in \R^k$ : data points 
- $g(x) : \R^k \to \R^l$ observation operator (not invertible, differentiable)
- $h(x) : \R^l \to \R$ constraint operator (not differentiable - not invertible)
- $\mathcal C : \{ l \in \R^l; h(l) = 0\}$ : constraint set 
- $\mathcal P(l) \triangleq \arg \min_{z \in \mathcal C} ||l - z||$ : projection operator



**Objective :** 

We want to sample from distribution $p(x)$ while respecting the following constraints $\E[h(g(x))] = 0$ and $\E[h(g(x))^2] \leq \epsilon$ using lagrangian formulation : 
$$
\min_q KL(q || p) + \lambda \E_{x \sim q}[h(g(x))] + \mu (\E_{x \sim q}[h(g(x))^2] - \epsilon)
$$
We introduce $z \in \R^l$ such that $z \in \mathcal C$





**Algorithm :**

The optimization problem we are doing is 
$$
\begin{aligned}
\min_{q(x)}\ & KL\bigl(q(x)\,\Vert\, p(x)\bigr) + X_{\mathcal C}(z) \text{ s.t. } \left\{
\begin{aligned}
\mathbb{E}_{q(x)}[g(x) - z] &= 0,\\
\mathbb{E}_{q(x)}[\|g(x) - z\|^2] &\le \epsilon_2.
\end{aligned}
\right.
\end{aligned}
$$
Were $X_{\mathcal C}(z)$ is $\infty$ is $z$ doens't respect the constraint. This leads to an optimisation / projection algorithm : 
$$
\begin{aligned}
x_{t+1} &= x_t 
+ \textcolor{red}{\Delta_t}\Big( 
    \nabla_x \log p(x_t) 
    - \nabla_x \big[\,\rho_t\,\|g(x_t) - z_t\|^2 
                     + \lambda_t^\top (g(x_t)-z_t)\big]
\Big)\\[0.4em]
z_{t+1} &=  \arg \min_{z \in \mathcal C} 
\big\|g(x_t) + \big(2\,\rho_t\big)^{-1} \lambda_t - z\big\|^2\\[0.4em]
\lambda_{t+1} &= \lambda_t 
+ \textcolor{red}{\eta}\;\mathbb{E}\big[g(x_t) - z_t\big]\\[0.4em]
\rho_{t+1} &=  \Big[\rho_t
+ \textcolor{red}{\eta}\Big(\mathbb{E}\big[\|g(x_t) - z_t\|^2\big] 
- \textcolor{red}{\varepsilon}\Big)\Big]_{+}
\end{aligned}
$$


**Question :** 

- I am really not sure how to justify ththe change, especially the relationship betwee $\epsilon$ and $\epsilon_2$. 

- Also the version over the batch is interesting but I have no idea how to justify it. 



The algorithm is globally : 

```
x_{t+1} = score(x_t) + lambda * \nabla_x E_{q(x)}[||x-z||^2] 
z_{t+1} = \argmin_z\inC W(x_{t+1}, z) # the closest distribution that respect the constraint
lambda_{t+11} = E_{q(x)}[||x-z||^2] 
```

