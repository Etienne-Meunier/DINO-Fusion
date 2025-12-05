# Optimisation with decomposition



### Framework 

- Cost function $f : \R^n \to \R$
- Observation operator $g : \R^n \to \R^m$
- Constraint set : $C_{set} = \{C(g(x)) = 0\}$  with $C : \R^m \to \R$

$$
\min_x f(x) \text{ s.t } C(g(x)) = 0
$$

### Naive option :

$$
\min_x f(x) \text{ s.t } C(g(x)) = 0 \to \min_x f(x) \text{ s.t } \{ x = z, C(g(z)) = 0 \}
$$

We can formulate an augmented lagrangian as : 

$$
L_a = f(x) + X_c(g(z)) + \frac{\rho}{2}\big [||x + \mu - z||^2 - ||\mu||^2\big ]
$$
Here $\{x, z, \mu \} \in \R^n$, this results in steps : 
$$
\begin{align*}
x^* &= \arg \min_x f(x) + \frac{\rho}{2}||x + \mu - z||^2 \to \text{No problem} \\
z^* &= \arg \min_z  X_c(g(z)) + \frac{\rho}{2}||x + \mu - z||^2  = P_c(x + \mu)\to \text{Projection} \\
\end{align*}
$$
The diffuculty here is that the projection might be compicated to do, it amounts to find $z^*$ close to $x+\mu$ such as $C(g(z^*)) =0$ which would requires to invert $C$ and $g$ or rely on some kind of optimisation.

### Option 1 :

We can choose to decouple differently, reformulating the constraint : 
$$
\min_x f(x) \text{ s.t } C(g(x)) = 0 \to \min_x f(x) \text{ s.t } \{ g(x) = z, C(z) = 0 \}
$$
Where $x \in \R^n$ and $z \in \R^m$ 
$$
\begin{align*}
L_a = f(x) + X_c(z) &+ \frac{\rho}{2}\big [||g(x) + \mu - z||^2 - ||\mu||^2\big ]
\end{align*}
$$

$$
\begin{align*}
x^* &= \arg \min_x\ f(x) + \frac{\rho}{2}||g(x) + \mu - z||^2 \to \text{Some problems} \\
z^* &= \arg \min_z\ X_c(z) + ||g(x) + \mu - z||^2 \to \text{Projection} \\
\end{align*}
$$

In this case the projection is easy to do because it's directly in $R^m$ but the update on $x$ mix $f(x)$ and $g(x)$ two functions that might have different regularity and that are complex to optimize. 

### Option 2 :

Another decomposition we can try is : 
$$
\min_x f(x) \text{ s.t } C(g(x)) = 0 \to \min_x f(x) \text{ s.t } \{x = z, g(z) = y, C(y) = 0 \}
$$
Where $\{x, z\} \in \R^n$ and $y \in \R^m$, we get a lagrangian as : 

$$
\begin{align*}
L_a = f(x) + X_c(y) &+ \frac{\rho_1}{2}\big [||x + \mu_1 - z||^2 - ||\mu_1||^2\big ]\\
&+ \frac{\rho_2}{2}\big [||g(z) + \mu_2 - y||^2 - ||\mu_2||^2\big ]
\end{align*}
$$
Where $\mu_1 \in \R^n$ and $\mu_2 \in \R^m$.

Which results in updates : 
$$
\begin{align*}
x^* &= \arg \min_x\ f(x) + \frac{\rho_1}{2}||x + \mu_1 - z||^2 \to \text{No problem} \\
z^* &= \arg \min_z\ \rho_1||x + \mu_1 - z||^2 + \rho_2||g(z) + \mu_2 - y||^2 \to \text{No problem} \\
y^* &= \arg \min_y X_c(y) + \frac{\rho_2}{2}||g(z) + \mu_2 - y||^2 \to \text{Projection}  \\
\end{align*}
$$
Then update on $\mu_1$ and $\mu_2$ are trivial. 

The interesting point of this approach is that we optimise separately $x$ and $z$ and that the projection is done in $R^m$ (so might be easier to do). 

- Find sources for that method 
- The optimisation on $z$ is maybe some kind of interpolation. 

