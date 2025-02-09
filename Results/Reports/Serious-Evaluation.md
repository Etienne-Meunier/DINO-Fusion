# Paper ICLR - 31/01/25



Run with constraint : `/tav0h83b/inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/20250130-165431.npy`

 [constraints_border_zero_gradient_zero_mean:20250130-165431.zip](../../../../../Downloads/constraints_border_zero_gradient_zero_mean:20250130-165431.zip) 

Run without constraint : `tav0h83b/inference/infesteps_1000/constraints_no_constraints/20250131-110120.npy`







### The constraint : 

![image-20250131105844577](./images/image-20250131105844577.png)

**Other modifications :** 

```python
# Delt with boundary issues at the bottom
generated_samples['toce.npy'][:,:,1, :] = generated_samples['toce.npy'][:,:,2, :]
generated_samples['soce.npy'][:,:,1, :] = generated_samples['soce.npy'][:,:,2, :]
generated_samples['ssh.npy'][:,1, :] = generated_samples['ssh.npy'][:,2, :]
```



## Larger - generation

We generate batches of 200 with seed 123. 



```python
model_path = /Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/
no_constraint = model_path + 'constraints_no_constraints/20250203-141645.npy'
constraint = model_path + 'constraints_border_zero_gradient_zero_mean/20250203-175158.npy'
```

Which we use to generate to clean (renormalised) version ('with -clean instead of .npy')





Formally, we train a denoising model $\epsilon_\theta(x_s, s)$ to predict the noise added to the data, where $\theta$ represents the neural network parameters and $s \in [0,S]$ the diffusion step. The training objective is:
$$
\begin{equation}
    \mathcal{L}(\theta) = \mathbb{E}_{s,x_0,\epsilon} [||\epsilon_\theta(x_s, s) - \epsilon||^2]
\end{equation}
$$
where $x_s = \sqrt{\bar{\alpha}_s}x_0 + \sqrt{1-\bar{\alpha}_s}\epsilon$ with $x_0$ sampled from our training data, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$, and $\{\bar{\alpha}_k = \prod_1^s \alpha_k \}_{s=1}^S$ is a fixed variance schedule decreasing from 1 to 0.

To generate new samples, we start from Gaussian noise $x_S \sim \mathcal{N}(0, \mathbf{I})$ and iteratively denoise using:
$$
\begin{equation}
    x_{s-1} = \alpha_s^{-\frac12}(x_s - \beta_s\epsilon_\theta(x_s, s)) + \sigma_s \mathbf{z}
\end{equation}
$$
where $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, $\beta_s = (1-\alpha_s) (1-\bar \alpha_s)^{-\frac12}$, and $\sigma_s$ is the standard deviation of the reverse process noise. This sampling procedure progressively transforms pure noise into realistic oceanic states.
