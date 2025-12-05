## Constrained Sampling Alternative 



Here we develop the alternative formulation of constrained sampling from Blanke where $z$ lives in a different space as $x$ :

The basis was to consider a constrained sampling with Lagrangian ($f : \R^n \to \R$) : 
$$
u(x, z, \mu) = f(x) + X_c(z) + \rho \mu^T(x-z)  + \frac{\rho}{2}||x-z||^2
$$
(we work with scaled dual variable as $\lambda = \mu / \rho$)

Here we consider function $g : \R^n \to \R^m$ which project the data into a different space (ex : density profile), we can then formulate the constraint as : 

$$
u(x, z, \mu) = f(x) + X_c(z) + \rho \mu^T(g(x)-z)  + \frac{\rho}{2}||g(x)-z||^2
$$
Where $x \in \R^n$,  $\mu \in \R^m$ , $z\in \R^m$ and $\rho \in \R$ . 

> [!NOTE]
>
> This correspond to the same problem but with a different set of constraint : 
>$$
>\min_x f(x) \text{ s.t } \{g(x) = z \text{ and } C(z) = 0\}
>$$


We can compute the update from ADMM (where $J_g : \R^n \to \R^{n\times m}$ is the jacobian of $g$)  : 

1. Update w.r.t $x$ :


$$
\begin{align}
\nabla_x u &= \nabla_x f + \rho J_g(x)^T \mu + \rho J_g(x)^T(g(x) - z)\\
 &= \nabla_x f + \rho J_g(x)^T (\mu + g(x) - z)
\end{align}
$$

```python
def g(x) : 
		return density_profile(x) 
	
 
u_dx = grad_f(x) + rho*torch.func.jvp(g, x,  (mu + g(x) - z))
```



2. Update w.r.t $\mu$ : 


$$
\nabla_\mu u = \rho (g(x) -  z)
$$

3. Update (projection for $z$) : 

$$
u(x, z, \mu) = f(x) + X_c(z) + \frac{\rho}{2}||g(x) -  z + \mu||^2 - \frac{\rho}{2} ||\mu||^2
$$

Thus : 
$$
\arg \min_z u(x, z, \mu) =  \frac{\rho}{2}||g(x) -  z + \mu||^2  + X_c(z) = P_c(g(x) + \mu)
$$


### Projections 



1. Mean density profile 

In the first constraint we imposed we choosed that the density profile $g(x)$ should be the same as the one diagnosed in data $\gamma \in \R^m$ : in this case the projection is trivial as there is only one $z$ that respect that which is $\gamma$ itself, in this case we can just replace $z$ with $\gamma$ and just run updates on $x$ and $\mu$

2. Gradient density profile : 

Now we would like to impose a constraint such that $\frac{dg}{dz}(x) \triangleq [g_{i+1}(x) - g_{i}(x)]$ (vertical gradient of the densit of the density profile) is the same as $\frac{d \gamma}{dz}$, how can I build my proximal operator ? 