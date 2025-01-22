# Results Analysis 



### 22 / 01 /25 - z87envpm



It seems like the model have trouble to predict "extreme values" like cold in the south : 

![image-20250122180749328](/Users/emeunier/Library/Application Support/typora-user-images/image-20250122180749328.png)

We can confirm on more data comparing the average temperature in the south and north block as we can see the north is always warmer than the south : 

![image](/Users/emeunier/Downloads/image.png)



![image (3)](/Users/emeunier/Downloads/image (3).png)

Although in the generated data it's not the case 

![image (1)](/Users/emeunier/Downloads/image (1).png)

##### Hypothesis 1 : issue with normalisation ? 

- If we take the data normalise are we still observing the same trend ? 
- If we compare them to the generation non-denormalised do we have something similar ? 



##### Hypothesis 2 : Diffusion bad at generating extremes ? 

Maybe the diffusion model is just generated states close to an average value everywhere for some reason ? 