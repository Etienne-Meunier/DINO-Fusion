# Results Analysis 



### 22 / 01 /25  First Naive test - z87envpm

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

On real data we can see the huge gap : 

![image-20250127113854487](./images/image-20250127113854487.png)

### 22 / 01 /25  Retraining with min-max normalisation - hxdnrm4i

After min-max normalisation the problem seem to be fixed : 

![image-20250127111330882](./images/image-20250127111330882.png)

However we still don't have a perfect north/south distribution :

![image-20250127110118907](./images/image-20250127110118907.png)

Especially on toce, the south block have very small variabnce and should be detached from the rest. 

Exploring real values for comparison : 

![image-20250127133540778](./images/image-20250127133540778.png)

We have some variability in the profile

![image-20250127114540310](./images/image-20250127114540310.png)

![image-20250127131129937](./images/image-20250127131129937.png)

![image-20250127131105867](./images/image-20250127131105867.png)

Solution Julie : tu prends un profil moyen de t-s et tu prédit S - 3

**Remaining issues :**

1.  <u>Stratification</u>

It seems like our base model is biased on the extreme values which is not great

![image-20250127131331684](./images/image-20250127131331684.png)

2. Data distribution at bottom : 

![image-20250127140702869](./images/image-20250127140702869.png)

 Generated data are two warm at z big. Why ? 

Firs, it seems to not be a obvious bug with normalisation as it is possible to get those values in data (a 1.0 in generation will correspond to the max for the depth) : 

![image-20250127142211200](./images/image-20250127142211200.png)

As expected generated fields never get above the max. We can see it in the non-normalized version (where no generated values are above 1) : 

![image-20250127142520433](./images/image-20250127142520433.png)



Although the issue is that we ar enoyt capturing the dynamic of the distribution well. 

