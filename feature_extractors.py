import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LBPExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.offsets = torch.tensor([
            [-1, -1], [-1,  0], [-1,  1],
            [ 0,  1], [ 1,  1], [ 1,  0],
            [ 1, -1], [ 0, -1]
        ], dtype=torch.long)
        self.out_dim = 256

    def forward(self, x):
        bs, _, H, W = x.shape
        gray = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
        pad = F.pad(gray, (1,1,1,1), mode='replicate')
        codes = torch.zeros(bs, 8, H, W, device=x.device)
        for i,(dy,dx) in enumerate(self.offsets):
            neigh = pad[:,:,1+dy:1+dy+H, 1+dx:1+dx+W]
            mask = (neigh >= gray).float().squeeze(1)
            codes[:,i] = mask
        weights = 2 ** torch.arange(8, device=x.device).view(1,8,1,1).float()
        lbp = (codes * weights).sum(dim=1).long()
        hist = torch.stack([
            (lbp == b).float().sum(dim=(1,2)) for b in range(256)
        ], dim=1)
        hist = hist / (hist.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return hist


class HOGExtractor(nn.Module):
    def __init__(self, cell_size=8, nbins=9, image_size=224):
        super().__init__()
        self.cell_size = cell_size
        self.nbins = nbins
        n_cells = (image_size // cell_size) ** 2
        self.out_dim = n_cells * nbins
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        bs, _, H, W = x.shape
        gray = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        ang = torch.atan2(gy, gx) % (2*math.pi)
        cs = self.cell_size
        mags = F.unfold(mag, kernel_size=cs, stride=cs)
        angs = F.unfold(ang, kernel_size=cs, stride=cs)
        bs2, _, n_cells = mags.shape
        mags = mags.transpose(1,2).reshape(-1, cs*cs)
        angs = angs.transpose(1,2).reshape(-1, cs*cs)
        bin_w = 2*math.pi / self.nbins
        idx = (angs / bin_w).long() % self.nbins
        hist = torch.zeros(mags.size(0), self.nbins, device=x.device)
        hist.scatter_add_(1, idx, mags)
        hist = hist.view(bs, n_cells * self.nbins)
        hist = hist / (hist.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return hist


class SIFTExtractor(nn.Module):
    def __init__(self, cell_size=8, nbins=8, sigma=1.0, image_size=224):
        super().__init__()
        self.cell_size = cell_size
        self.nbins = nbins
        coords = torch.arange(cell_size) - (cell_size - 1)/2
        xx, yy = torch.meshgrid(coords, coords, indexing='ij')
        gauss = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).view(1, cell_size*cell_size, 1)
        self.register_buffer('gauss', gauss)
        n_cells = (image_size // cell_size) ** 2
        self.out_dim = n_cells * nbins
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
        ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        bs, _, H, W = x.shape
        gray = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        ang = torch.atan2(gy, gx) % (2*math.pi)
        cs = self.cell_size
        mags = F.unfold(mag, kernel_size=cs, stride=cs) * self.gauss
        angs = F.unfold(ang, kernel_size=cs, stride=cs)
        bs2, _, n_cells = mags.shape
        mags = mags.transpose(1,2).reshape(-1, cs*cs)
        angs = angs.transpose(1,2).reshape(-1, cs*cs)
        bin_w = 2*math.pi / self.nbins
        idx = (angs / bin_w).long() % self.nbins
        hist = torch.zeros(mags.size(0), self.nbins, device=x.device)
        hist.scatter_add_(1, idx, mags)
        hist = hist.view(bs, n_cells * self.nbins)
        hist = hist / (hist.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return hist
