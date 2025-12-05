$$
\newcommand{\E}{\mathbb E}
$$



# Sampling on density profiles 

On pourrait définir des "profils de points" dans $R^2$ comme $(x_1, x_2)$ et chaque "sample" est consitué de $K$ = 10 points, on a $N$ profils en tout dans l'entrainement. Ces points respectent majoritairement une contrainte (ex : $C_1 : x_1^2 + x_2^2 = r$ ou $C_2 : x_1 <  x_2$) et on veut respecter cette contrainte de la même manière dans la génération. 

1. On peut estimer dans les données le respect de chaque contrainte : 

$$
\mathbb E_{p_{data}}[x_1^2 + x_2^2] = \frac{1}{KN}\sum_i^N \sum_k^K(x_{1,k}^2 + x_{2,k}^2) =\tilde r \quad \mathbb E_{p_{data}}[((x_1^2 + x_2^2) - \tilde r)^2] = \tilde \sigma
$$

2. Quand on fait la génération on impose ces contraintes avec égalité + inégalité de Lagrange 



We try to generate a random variable $x \in \R^d$ for which we have an approximation of $\nabla \log p(x)$, however we want the variable to respect some constraints define in an observation space $g(x) \in \R^k$. Thoses constraints should be respected by the generated data in the same way as they are in the original data (so not strictly for every sample) :  
$$
\begin{align}
\mathcal L(x, z, \lambda, \mu) &= \log p(x) + \lambda^T \E[g(x) - z - \kappa] + \mu \E[||g(x) - z||^2 - \sigma] + X_{\mathcal C}(z) \\
&=  \log p(x) + X_{\mathcal C}(z) + \mu \E[||g(x) - z + \alpha||^2] - \mu ||\alpha||^2  - \mu \sigma - \lambda^T \kappa\text{ with }\alpha = \frac{1}{2 \mu}\lambda

\end{align}
$$
where $\lambda \in R^k$ and $\mu \in \R^+$ and $\kappa \in \R^k$, $\sigma \in \R$ is diagnosed from the data : 
$$
\kappa = \E_{x \sim p_{data}}[g(x)] \quad \sigma = \E_{x \sim p_{data}}[||g(x) - z||^2]
$$
We can do the sampling in four steps : 
$$
\begin{align}
\text{Sampling : }& \partial_t x= \nabla_x \mathcal L(x, z, \lambda, \mu)\\
\text{Projection : }& z^* = \arg \min_z  X_{\mathcal C}(z) + \mu \E[||g(x) - z + \alpha||^2]\\
\text{Constraint $\mu$ : }& \partial_t \mu = \big[ \E[||g(x) - z||^2 - \sigma] \big]_+\\
\text{Constraint $\lambda$ : }& \partial_t \lambda = \E[g(x) - z + \kappa]
\end{align}
$$

### Constraints : 

1. Average profile : 

A first set of constraint is to define an average profile $g^* \in \R^k$ which the data should follow. In this case the projection is trivial (just returning $z = g^*$)

2. Isotonic function 

If we want $g(x)_{k+1} > g(x)_k\ \forall k$  we can project the density profile using an isotonic projection. Then $z$ is the closest monotonically increasing profile to $g(x)$.



## Toy example 

I want to display this methods on a toy example. First I generate a dataset of points in 2D that lie on a circle of radius $r$. All the points in the training set lie on the radius with a tiny noise (gaussian) so that $E_{p_{data}}[||x||] = r$ and $E_{p_{data}}[(||x|| - r)^2] = \sigma$. Then we train a diffusion model on it so that we can sample new point.  In this setup the observatio is $g(x) = ||x||$ and the constraint is $h(g(x)) = r$

- Step 1 : we do sampling without constraint and measure $E_{p_{gen}}[||x||]$ and $E_{p_{gen}}[(||x|| - r)^2]$
- Step 2 : sampling with constraint with different $\sigma_c$ and see if we reach the required variance

Another constraint I would like to show is that $x_1 \leq x_2$ in this setup : I guess we have two ways to do this 

<u>Way 1 :</u> Consider $g(x) = x$ and $z = \text{isotonic}(g(x))$ returning the monotically increasing profile the closest to $g(x)$ for example using `spicy.isotonic`. In this case we can compute $\sigma$ in a similar fashion as before : the distance of $g(x)$ and it's closest monotically increasing profile.

<u>Way 2 :</u> Consider $g(x) = (x_2 - x_1)$ compare that $\E[g(x)] < 0$ with an extra loss penalizing $\E[[g(x)]_+ ^2]$  (the square of the positive part of g(x)) on which we can compute $\sigma$ on the data too.





Q : Comment imposer ces contraintes quand on a une projection mais pas une optimisation ? 

Q : Comment générer des exemples pour vérifier la contrainte de croissance ? 





TODO : 

- Faire le lien avec  "approximate I-projection"
- Proposer des contraintes avec matrices de covariances