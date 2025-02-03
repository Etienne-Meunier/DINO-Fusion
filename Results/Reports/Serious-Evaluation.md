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

