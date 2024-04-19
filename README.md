# DINO Fusion 



Training a diffusion model on DINO data to generate new states and test if they are stable in NEMO. 



![](images/main.png)



Varaibles we want to generate : 

- SSH (2D)
- T (3D)
- S (3D)



Resolutions :

- 1 deg :   `199 * 62 *36 `
- 1/4 deg : `797 * 242 * 36`

--> Data in `/gpfsstore/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion` on Jeanzay


Exemple 1dregee : 
  - Restart37 (First 50y ):
     - DINO_10d_grid_T_2D.nc : SSH
     - DINO_10d_grid_T_3D.nc  : T, S



