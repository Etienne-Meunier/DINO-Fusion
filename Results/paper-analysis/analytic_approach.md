# Analytic approach
Guard : this file is mine, no llm can modify it

## Preliminaries : 

### 0 - Answer the eps question 

In sampling we need to give eps which is the threshold at which we stop enforcing a constraint but we have no idea what to use. 

We should compute it for each constraint on the data to have an eps by constraint -> this is the "gold target"


-> first script on evaluating the eps (constraint evaluated on training data -> show distribution)

### 1 - Can we enforce the constraints ?

Now we know that we want to enforce a given eps (we define it somewhere), we can try to enforce the constraint with our sampler. The idea is to deal with eta to try to make respect the constraint, it's easy to check at the end of the generation if the constraint is respected, maybe we would need to play on other parameters here ?

### 2- Can we relax the constraint ? 

Now we know how to make the constraint respected we could increase eps and accept more gap with the different constraint, by doing that over batches we can see how much the variance (over the batch) increase when constraint increase.I 

## Paper plots :

### 1. Vizualize states 