# Message DINO-Fusion 



We can generate states that are *similar (1)* to the data but that *doesn't respect physical constraints (2)*. This has an impact on the way the model integrate those states and how they can be used in a forecasting pipeline. 



(1) similar : taking into account a set of metrics implictly used to train the diffusion model 

(2) physical constraint : for instance stratification



1. Show that the generated states (without constraints) are similar (in some metrics (1) sense) to the data but don't respect physical constraints (2) 
2. Show that constrained states are less similar in (1) but are more respectful in (2)
3. Show that the constrained states are more stable when given to the integrator than the unconstrained states 



### Problems : 



#### Metrics for (1) 

-  We need to find the most neutral possible metric (no physical information) to compare the physi

