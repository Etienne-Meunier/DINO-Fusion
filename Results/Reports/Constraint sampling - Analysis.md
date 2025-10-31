# Constraint sampling - Analysis 



One of the issues in the density constraint is that it is not valid on noisy states for the upper layers, indeed adding gaussian noise to the input break the constraint for noisy state, thus I am not sure if it's working well, especially if we use "hard" projection methods on it : 

![image-20251022173548887](/Users/emeunier/Library/Application Support/typora-user-images/image-20251022173548887.png)



Options : 

1. Implement ADMM/Blank and check if the constraint is valid in this case 
2. Check with an implementation like tweedy if it works well ? 
3. Should we impose the constraint on each profile instead ? 