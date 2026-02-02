# Generated files profiles 

http://localhost:8888/notebooks/Results/notebooks/plot-density-profiles.ipynb

If we look at a field from data and the cluster the density we get regions like that : 

![image-20260114161201300](./images/image-20260114161201300.png)





## Generated

![image-20260114161438077](./images/image-20260114161438077.png)

![image-20260114161519669](./images/image-20260114161519669.png)

![image-20260114161541324](./images/image-20260114161541324.png)

![image-20260114161643624](./images/image-20260114161643624.png)

-> Constraints manage to keep the structure of the state (cluster map over density profiles), which is great





# Generate with isotonic constraint 

![image-20260119174819550](./images/image-20260119174819550.png)



![image-20260119174828214](./images/image-20260119174828214.png)

Good point : density at bottom is good, less good : density at surface is shifted

![image-20260120124636589](./images/image-20260120124636589.png)

Le problème qu'on a sur la génération de profils isotoniques c'est 

	- La densité a la surface paraît trop haute 
	- On arrive pas a faire respecter la croissance monotone 
	- C'est peut etre a cause des hyperparametres mais on sait pas 

-> Est ce qu'on devrait pas continuer a travailler sur l'algorithme / la selection des hyperparametres ? 
