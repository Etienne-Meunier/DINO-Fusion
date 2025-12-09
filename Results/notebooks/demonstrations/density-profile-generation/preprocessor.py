import torch
from ShapeChecker import ShapeCheck

class Preprocessor :
    def __init__(self, shapes, mean, std, levels) : 
        """
            mean (1, channels)
        """
        self.sc = ShapeCheck(shapes, 'n k i j')
        self.mean = mean 
        self.std = std
        self.levels = levels 

    def normalize(self, data) :
        """
           data (b, channels)

        """
        return (data - self.mean)/self.std

    def unnormalize(self, data) :
        """
           data (b, channels)

        """
        return (data*self.std) + self.mean
    
    def concat(self, toce, soce) : 
        assert toce.shape[-1] == self.levels
        data = self.sc.rearrange([toce, soce], 'm b k -> b (m k)')
        self.sc.update(data.shape, '_ c')
        return data
    
    def unconcat(self, data) : 
        toce = data[:, :self.levels]
        soce = data[:, self.levels:]
        return toce, soce
    
    def flatten(self, field) : 
        return self.sc.rearrange(field, 'n k i j -> (n i j) k')

    def unflatten(self, field) : 
        return self.sc.rearrange(field, '(n i j) k -> n k i j')
    
        
    def forward(self, toce, soce) : 
        data = self.concat(toce, soce)
        return self.normalize(data)
    
    def backward(self, data) : 
        data = self.unnormalize(data)
        toce, soce = self.unconcat(data)      
        return toce, soce



def get_density_at_surface_tensor(thetao, so, tmask=None):
    """
    Compute potential density referenced at the surface using PyTorch tensors.

    Parameters:
        thetao (torch.Tensor): Temperature tensor - (t, z, y, x).
        so (torch.Tensor): Salinity tensor - (t, z, y, x).
        tmask (torch.Tensor): Mask tensor - (t, z, y, x).

    Returns:
        torch.Tensor: Potential density referenced at the surface.
    """
    if tmask is None : 
        tmask = torch.ones_like(thetao)

    # Constants
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0

    # EOS coefficients
    EOS000 = 8.0189615746e02
    EOS100 = 8.6672408165e02
    EOS200 = -1.7864682637e03
    EOS300 = 2.0375295546e03
    EOS400 = -1.2849161071e03
    EOS500 = 4.3227585684e02
    EOS600 = -6.0579916612e01
    EOS010 = 2.6010145068e01
    EOS110 = -6.5281885265e01
    EOS210 = 8.1770425108e01
    EOS310 = -5.6888046321e01
    EOS410 = 1.7681814114e01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e01
    EOS120 = 6.1548258127e01
    EOS220 = -6.0362551501e01
    EOS320 = 2.9130021253e01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e01
    EOS130 = -3.3449108469e01
    EOS230 = 1.9717078466e01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e01
    EOS101 = -4.2549998214e01
    EOS201 = 5.0774768218e01
    EOS301 = -3.0938076334e01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01

    # Tensor computations
    zt = thetao * r1_T0  # temperature
    zs = torch.sqrt(torch.abs(so + rdeltaS) * r1_S0)  # square root salinity
    ztm = tmask.squeeze()

    

    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    zn2 = ((EOS022 * zt + EOS112 * zs + EOS012) * zt + (EOS202 * zs + EOS102) * zs + EOS002)
    zn1 = (((((EOS041 * zt + EOS131 * zs + EOS031) * zt + (EOS221 * zs + EOS121) * zs + EOS021) * zt
            + ((EOS311 * zs + EOS211) * zs + EOS111) * zs + EOS011) * zt
            + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs + EOS001))
    zn0 = (
        (
            (
                (
                    (
                        (EOS060 * zt + EOS150 * zs + EOS050) * zt
                        + (EOS240 * zs + EOS140) * zs
                        + EOS040
                    )
                    * zt
                    + ((EOS330 * zs + EOS230) * zs + EOS130) * zs
                    + EOS030
                )
                * zt
                + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs
                + EOS020
            )
            * zt
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110)
            * zs
            + EOS010
        )
        * zt
        + (
            ((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs
            + EOS100
        )
        * zs
        + EOS000
    )
    rhop = zn0 * ztm  # potential density referenced at the surface
    return rhop
