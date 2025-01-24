# Results Analysis 



### 22 / 01 /25 - z87envpm



It seems like the model have trouble to predict "extreme values" like cold in the south : 

![image-20250122180749328](./images/image-20250122180749328.png)

We can confirm on more data comparing the average temperature in the south and north block as we can see the north is always warmer than the south : 

![image](./images/image.png)



![image-20250122185129158](./images/image-20250122185129158.png)

Although in the generated data it's not the case 

![image (1)](./images/image (1).png)

##### Hypothesis 1 : issue with normalisation ? 

- If we take the data normalise are we still observing the same trend ? 
- If we compare them to the generation non-denormalised do we have something similar ? 





![image-20250124135657647](./images/image-20250124135657647.png)

##### Hypothesis 2 : Diffusion bad at generating extremes ? 

Maybe the diffusion model is just generated states close to an average value everywhere for some reason ? 

![image-20250124141436005](./images/image-20250124141436005.png)

![image-20250124141442083](./images/image-20250124141442083.png)

Ok seems to be that we have a normalisation problem. We just needed to bring the data between -1 and 1 for training. 

After min-max normalisation the problem seem to be fixed : 

![image-20250124180630861](./images/image-20250124180630861.png)

