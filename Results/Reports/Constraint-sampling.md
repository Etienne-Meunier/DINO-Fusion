# Problem setup 

In this work we aim to train a generative model on a distribution of states generated from an oceanographic numerical simulation. Using the DINO configuration with parameter $\gamma$, we generate a sequence of states at the $\frac14 \degree$ resolution representing the evolution of the simulation over 50 years. Each of those states are discretized on a 3D grid of size $Z \times W \times H$ where $Z$ is the vertical and $W, H$ the spatial number of cells. In this paper we use indices $(k, i, j)$ to reference a cell over this grid, $k$ represents depth level and $(i,j)$ horizontal coordinates.

Those states present a large number of dynamic variables but we focus in this on two prognostic variables central for ocean large scale dynamics : temperature $Q$ and salinity $S$. We normalise each of them by vertical level before concatenating them to form of state $X = ( Q || S )$. 

Our dataset is thus a sequence of state $\{X_t\}_{t \in \{0, \dots, T\}}$. In the next section we describe the training of a generative model $p_\theta(X | \gamma)$ approximating this true distribution of simulation.

**Generative process**

Our generative model is based on denoising diffusion probabilistic models \cite{DDPM} we train a denoising model $\epsilon_\theta(x, l)$ (denoting $l$ as the diffusion step to avoid confusion with simulation time $t$) using a classical denoising approach to approximate the score function of the state distribution with the following objective:
$$
\begin{equation}
θ = \arg \min_θ \mathbb{E}_{l,x_0,\epsilon} [||\epsilon_\theta(\sqrt{\bar{α}_l}x_0 + \sqrt{1-\bar{α}_l}\epsilon, l) - \epsilon||^2]
\end{equation}
$$
Sampling is performed using a Langevin-like dynamics:
$$
\begin{align}
&x_L \sim N(0, \mathbf{I})\\ \nonumber
&x_{l-1} = \frac{1}{\sqrt{\bar \alpha_l}}(x_l - \frac{1-\bar \alpha_l}{\sqrt{1-\bar \alpha_l}}\epsilon_\theta(x_l, l)) + \sigma_l \mathbf z 
\end{align}
$$
Where $l \sim \mathcal U(0, L )$,  $\epsilon$ and $\mathbf z$ are sampled from multivariate normal distribution and $\bar \alpha_l$ and $\sigma_l$ are dependent on the variance schedule (see details in \cite{DDPM}).

**Physical constraints**

As we will show in the results section, states generated from diffusion models do not respect necessary physical constraints, we could introduce a regularisation loss during training or architectural constraint to addess this problem, although this would require re-training the model. In this work we leverage a guiding process to enforce the constraints at sampling time.

We define a regularisation enforcing a constraint on the 3D field as $C(x) : \R^{Z \times W \times H} \to \R$  and then modify the sampling step exploiting the gradient of that constraint scaled by a given $\kappa(l)$ :
$$
\begin{align}
&x_{l-1} = \frac{1}{\sqrt{\bar \alpha_l}}(x_l - \frac{1-\bar \alpha_l}{\sqrt{1-\bar \alpha_l}}\epsilon_\theta(x_l, l)) - \kappa(l) \nabla C(x_l) + \sigma_l \mathbf z
\end{align}
$$
<u>Constraint on hydrostatic balance :</u>

An important constraint for oceanic state stability is to respect stratification (i.e density decrease with depth), in order to enforce this constraints in our generated states we formulate the following regularisation enforcing that the average spatial values for each vertical level should be close to the ones observed in oceanic simulation from the dataset   :
$$
\begin{equation}
C(x) = \frac{N}{2} \sum_{k} ( \mu_k - \frac{1}{N}\sum_{i, j} x_{ijk})^2
\end{equation}
$$
with $\mu_k$ the mean value for the vertical layer $k$ computed over the training dataset (in our case $\mu_k = 0$ as training data have been normalised) and N the number of horizontal cells. This constraints leads to an update applied to each cell :
$$
\begin{equation}
\partial_{i,j,k} C = \frac1N\sum_{i', j'} x_{i'j'k} - \mu_k
\end{equation}
$$
This regularisation is acting as a soft constraint on the mean vertical profile. We can balance between variability and constraint enforcement by adjusting $\kappa(l)$.
