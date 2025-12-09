import torch.nn as nn
import torch 

class Block(nn.Module) : 
    def __init__(self, in_ch, out_ch, activation=nn.ReLU, skip_co=False, norm=nn.Identity) : 
        super().__init__()
        self.skip_co = skip_co
        self.proj = nn.Linear(in_ch, out_ch, bias=True)
        self.act = activation()
        self.norm =  norm(out_ch)
    
    def forward(self, x) : 
        r = self.norm(self.act(self.proj(x)))
        if self.skip_co : 
            r = r + x
        return r
    
class Net(nn.Module) : 
    def __init__(self, in_ch) : 
        super().__init__()
        m =  256
        norm = nn.Identity
        act = nn.SELU
        blocks = [Block(in_ch+1, m, norm=norm, activation=act),
                       Block(m, m, skip_co=True, norm=norm, activation=act),
                       Block(m, m, skip_co=True, norm=norm, activation=act),
                       Block(m, in_ch, activation=nn.Identity)]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x, t) : 
        
        return self.blocks(torch.cat([x, t], dim=1))