$$
\newcommand{\E}{\mathbb E}
\newcommand{\V}{\mathbb V}
\newcommand{\C}{\mathcal C}
\DeclareMathOperator{\Tr}{Tr}
$$

# Variance vs ADMM form 

We have a distribution $p(x)$ define on a r.v $x \in \R^d$ and a $g : \R^d \to \R^k$. 

$h : \R^k \to \R$ represents a constraint and we define the set $\mathcal C : \{x ; h(g(x)) =0 \}$

We want to do a sampling as 
$$
\min_{q} KL(q || p) + \rho \E_{q}[h(g(x))^2] \text{ s.t } {\E_{q}[h(g(x))]} =0
$$
Normally we could solve that by defining a lagrangian and doing grandient descent-ascent (chamon approach) in that way : 
$$
\begin{align*}
x_{t+1} =& x + \nabla_x \log p(x) - \nabla_x [\rho h(g(x))^2 + \lambda^T h(g(x))] \text{ - Sampling} \\
\lambda_{t+1} &= \lambda + (\E_q[h(g(x))])

\end{align*}
$$
*With projection :*

And now the issue we have is that $h$ is not differentiable. Although we have access to a proximal operator $\mathcal P(g(x)) \triangleq \arg \min_{z \in \mathcal C} ||g(x) - z||^2$ . 

Thus we introduce a additional variable $z \in \C$ and formulate the problem as : 
$$
\min_{q(x, z)} KL(q(x) || p(x)) + \E[X_{\mathcal C}(z)] + \rho \E_{q(x,z)}[||g(x) - z||^2] \text{ s.t } {\E_{q(x, z)}[g(x)-  z]} =0
$$
Which leads to a lagrangian : 
$$
\min_{q(x, z)} KL(q(x) || p) + \E_{q(z)}[X_{\mathcal C}(z)] + \rho \E_{q(x,z)}[||g(x) - z||^2] + \lambda^T (\E_{q(x, z)}[g(x) -  z])
$$
And then the update will be 
$$
\begin{align}
x_{t+1} =& x + \nabla_x \log p(x) - \nabla_x [\rho ||g(x) - z||^2 + \lambda^T (g(x)-z)] \text{ - Sampling} \\
z_{t+1} &= \arg \min_{z \in \mathcal C} \rho ||g(x) - z||^2 + \lambda^T (g(x)-z) =  \arg \min_{z \in \mathcal C} ||g(x) + (2\rho)^{-1} \lambda - z||\\
\lambda_{t+1} &= \lambda_t + \eta \E[g(x) - z]

\end{align}
$$


**<u>Now in double constraint twist :</u>** 

Let's say instead I want to optimize : 
$$
\min_{q} KL(q || p) \quad \text{ s.t } {\E_{q}[h(g(x))]} =0 \text{ and }\E_{q}[h(g(x))^2] \leq \epsilon
$$
With variable split it would lead to : 
$$
\min_{q(x, z)} KL(q(x) || p(x)) + \E[X_{\mathcal C}(z)] \quad \text{ s.t } {\E_{q(x, z)}[g(x)-  z]} =0 \text{ and } \E_{q(x,z)}[||g(x) - z||^2] < \epsilon
$$
And a Lagrangian : 
$$
\min_{q(x, z)} KL(q(x) || p) + \E_{q(z)}[X_{\mathcal C}(z)] + \rho (\E_{q(x,z)}[||g(x) - z||^2] - \epsilon) + \lambda^T (\E_{q(x, z)}[g(x) -  z])
$$
Which leads to exactly the same update steps and add one on $\rho$ as : 
$$
\rho_{t+1} =  \big [\rho + \eta \E_{q(x,z)}[||g(x) - z||^2] - \epsilon \big ]_{+}
$$
