import numpy as np
import torch.nn as nn

def get_1d_sincos_pos_embed(emb_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(emb_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, emb_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(emb_dim, pos):
    assert emb_dim % 2 == 0
    omega = np.arange(emb_dim // 2, dtype=float)
    omega /= emb_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class ProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16,
        tubelet_size=1,
        in_chans=8,
        emb_dim=384,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(
            in_chans,
            emb_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x