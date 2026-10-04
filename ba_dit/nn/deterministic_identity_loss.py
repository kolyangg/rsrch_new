"""The same identity objective with deterministic bilinear face alignment."""
from torch.nn import functional as F


def aligned_crop(image, grid):
    """Equivalent to grid_sample(border, bilinear, align_corners=True).

    Grid coordinates are fixed training landmarks. PyTorch's deterministic
    gather backward avoids grid_sample's nondeterministic CUDA atomics.
    """
    batch,channels,height,width=image.shape
    x=((grid[...,0]+1)*(width-1)/2).clamp(0,width-1)
    y=((grid[...,1]+1)*(height-1)/2).clamp(0,height-1)
    x0,y0=x.floor().long(),y.floor().long()
    x1,y1=(x0+1).clamp_max(width-1),(y0+1).clamp_max(height-1)
    wx,wy=x-x0,y-y0
    def gather(xx,yy):
        index=(yy*width+xx).flatten(1)[:,None].expand(-1,channels,-1)
        return image.flatten(2).gather(2,index).reshape(batch,channels,*grid.shape[1:3])
    return (gather(x0,y0)*((1-wx)*(1-wy))[:,None] +
            gather(x1,y0)*(wx*(1-wy))[:,None] +
            gather(x0,y1)*((1-wx)*wy)[:,None] +
            gather(x1,y1)*(wx*wy)[:,None])


def identity_loss(vae, recognizer, velocity, case, reference_embedding):
    clean=case['noise'].float()-case['timestep'].float().reshape(-1,1,1)*velocity
    clean=clean.transpose(1,2).reshape_as(case['clean'])
    decoded=vae.decode(clean.to(vae.dtype)).float()
    aligned=aligned_crop(decoded,case['grid']).clamp(-1,1)
    embedding=F.normalize(recognizer(aligned),dim=-1)
    return (1-(embedding*reference_embedding).sum(-1)).mean()
