Here I want to try something new, try the impact on the constraint on data assimilation

The thing is that because the a-posteriori integration is messing up the state if it's not stratified, then if we start from a good ssh we would be get it messed up when we do assimilation. This is what we want to try. 

0) Do we have pattern 

Mise en place : 

    - In data do we have pattern of SSH or SST with respect to season ? 
    - Are those pattern accompanied with a deeper pattern (lower layer) ? 

Products : 

a. Figure with patterns and their associated deeper pattern 
b. Is SSH a good fit for pattern ? Is SST a good fit ? 

SSH would be the easiest for later phase this way it doesn't mix with the rest of the state


1) Can we assimilate data ?  

Mise en place : 

Here we take a fix sst from a data point and try to generate states with this fixed sst, for which we have two approach : 
    - Basic one : projection starting from the fixed sst we keep it fixed through the generation as we do for border 0 
    - Basic noisy one : we do the same but with noisy sst (as we did with NL earlier)

We then have approaches with nudging with a loss but it's a bit more cumberstone to do and tune so let's try those two first 

We generate with no constraints, huber and TS. Normally all should be doable as TS is fixing the mean which is fix through seasons. 

Evaluation : 

a. The sst should be good as we fixed it -> this is a first sign if we have a bug 
b. Looking at other factors (distribution of temperature in lower layers for instance) we can check if conditioning on sst have impacted the rest of the state accordingly ? 
c. How is the stratification for all state, do we still get a good stratification with huber and TS and a bad one for none ? 

Products : 

- A figure showing that the states are impacted from assimilation of sst 
- The density profile for all states 

At this state we decide which assimilation method we use (if basic works then we take this one)

2) How does it impact the integration ? 

Starting with those 5 (data, none, border0, border 0 + huber, border0 + TS) states as initial state and from the right season (summer / winter) we run 2w integration with daily outputs. We compare the resulting states, if evd destroyed the states for none then we know that the assimilation have not impacted this "in-practice" states.

So we do the run and then compare the sst of the integration with the one of the model initialized with data (we cannot use the reference here as the trajectory will be different). We can show the ssh after the 2w along with a graph with the difference between data ssh and model ssh (which should be 0 at initial time) and then we will come up with other plots

