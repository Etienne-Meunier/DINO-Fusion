$$
\newcommand{\E}{\mathbb E}
\newcommand{\V}{\mathbb V}
\DeclareMathOperator{\Tr}{Tr}
$$

# Variance vs ADMM form 

**<u>ADMM form :</u>** 

Starting from the initial problem
$$
\min KL(p || q) \text{ s.t } {\E_q[g(x)] = \mu}
$$
We can define an optimization form :
$$
L(q, z, x) = KL(p || q) + \lambda^T (\E_q[g(x)] - \mu) + \rho(||\E_q[g(x)] - \mu||^2)
$$
Where $\lambda \in \R^k$ and $\rho \in \R$. $\lambda$ is a Lagrangian multiplier and $\rho$ is a fix scalar. 

<u>Variance form :</u> 
$$
\min KL(p || q) \text{ s.t } {\E_q[g(x)] = \rho; \quad \E_q[\Tr((g(x) - \mu)(g(x) - \mu)^T] < \epsilon}
$$

$$
L(q, z, x) = KL(p || q) + \rho (\E_q[\Tr((g(x) - \mu)(g(x) - \mu)^T)) + \lambda^T (\E_q[g(x)] - \mu)
$$

With $\mu \in \R$ and $\lambda \in \R^K$. 

Because of trace cyclic property:
$$
\text{tr}(\epsilon \epsilon^T) = \text{tr}(\epsilon^T\epsilon) = ||\epsilon||^2
$$

$$
\begin{align*}
L(q, z, x) &= KL(p || q) + \rho (\E_q[||g(x) - \mu||^2]) + \lambda^T (\E_q[g(x)] - \mu)\\
&= KL(p || q) + \rho (||\E_q[g(x)] - \mu||^2) + \lambda^T (\E_q[g(x)] - \mu) + \rho \Tr (\V_q[g(x)])
\end{align*}
$$

*Optimisation the variance form :* 

If we want to optimise this objective we can take the following steps.
$$
x_{t+1} = x_t + \nabla_x \log p(x) + \nabla_x(\rho (\E_q[||g(x) - \mu||^2]) + \lambda^T (\E_q[g(x)] - \mu)) \text{     Sampling}
$$

$$

$$












#### Proofs on the trace (bias / variance tradeoff) :

<u>Let's say $g(x) \in \R$ :</u> 
$$
\begin{align*}
\E[||g(x) - \mu||^2] &= \E[(g(x) - \mu)^2] = \E[g(x)^2] - 2\mu\E[g(x)] +\mu^2 \\
&= \E[g(x)]^2 - 2\mu\E[g(x)] +\mu^2 + (\E[g(x)^2] - \E[g(x)]^2)\\
&= (\E[g(x)] - \mu)^2  + \text{var}[g(x)]
\end{align*}
$$
<u>If $g(x) \in \R^k$ :</u> 
$$
\begin{align*}
\E[||g(x) - \mu||^2] &= \E[(g(x) - \mu)^T(g(x) - \mu)] =\E[g(x)^Tg(x)] - 2\mu^T\E[g(x)] + \mu^T \mu \\
&=\E[g(x)]^T\E[g(x)] - 2\mu^T\E[g(x)] + \mu^T \mu + (\E[g(x)^Tg(x)] - \E[g(x)]^T\E[g(x)])\\
&=||\E[g(x)] - \mu||^2 + \Tr(Cov[g(x)])
\end{align*}
$$

$$
\begin{align}
Cov(X) &= \E[(X- \E[X])(X -\E[X])^T] = \E[XX^T]  - 2 \E[X]\E[X]^T + \E[X]\E[X]^T \\
&= \E[XX^T]  -  \E[X]\E[X]^T\\
\Tr((Cov(X)))&= \E[\Tr(XX^T)] - \Tr(\E[X]^T\E[X]) = \E[X^TX] - \E[X]^T\E[X] \\
\end{align}
$$

