import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Embedder import get_embedder
from torch import Tensor

class NeRF(nn.Module):
    def __init__(self,D=8,W=256,input_ch=3, input_ch_views=3, output_ch=4,use_viewdirs=True,skips=4):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_dir_ch = input_ch_views
        self.output_ch = output_ch
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        self.linears_layers = nn.ModuleList(
            [nn.Linear(input_ch,W)]+[nn.Linear(W,W) if i not in self.skips else nn.Linear(W+input_ch,W) for i in range(D-1)])

#第九层加入视角
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W ,W//2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W,W)
            self.alpha_linear = nn.Linear(W,1)
            self.rgb_linear = nn.Linear(W//2,3)
        else:
            self.output_linear = nn.Linear(W,output_ch)

    def forward(self,x):

        input_ch ,input_dir_ch = torch.split(x,[self.input_ch,self.input_dir_ch],-1)
        x1 = input_ch
        for i ,layer in enumerate(self.linears_layers):
            x1 = F.relu(layer(x1))
            if i in self.skips:
                x1 = torch.cat([input_ch,x1],-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(x1)
            feature = self.feature_linear(x1)
            h = torch.cat([feature,input_dir_ch],-1)

            for layer in self.views_linears:
                h = F.relu(layer(h))
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb,alpha],-1)
        else:
            outputs = self.output_linear(x1)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples