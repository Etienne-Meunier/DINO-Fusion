# Conditioning and Data-Assimilation



The goal of this set of experiments is to stufy if we can introduce conditioning over the state or a part of the state to do some kind of data-assimilation using our generative model. 

![image-20251020142216884](./images/image-20251020142216884.png)

## Temperature histograms 



1. Data 

![image-20251020142355064](./images/image-20251020142355064.png)

2. Generation without constraints (Just border 0)

![image-20251020163426017](./images/image-20251020163426017.png)

3. Generation with conditioning + no constraints 

![image-20251020170215707](./images/image-20251020170215707.png)



4. GradientZeroMeanConstraint

![image-20251021135046105](./images/image-20251021135046105.png)

On a une bonne génération sur certaines valeurs initiales mais sur d'autre un état assez bizarre 


