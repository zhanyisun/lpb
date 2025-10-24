import torch
import hydra
import dill
from einops import rearrange
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class ResNetEncoder(torch.nn.Module):
    def __init__(self, policy_ckpt_path, view_names, ):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path
        self.view_names = view_names
        self.emb_dim = 512
        self.latent_ndim = 2
        self.name = 'resnet'

        with open(self.policy_ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
            cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(payload['cfg'], output_dir='debug_obs_encoder')
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.to(device)
        policy.eval()

        self.obs_encoder = {}
        for view_name in self.view_names:
            self.obs_encoder[view_name] = policy.obs_encoder.obs_nets[view_name].backbone
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        del workspace
        del policy
        torch.cuda.empty_cache()

    def forward(self, x):
        view_embs = {}
        for view_name in self.view_names:
            imgs = x[view_name]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            imgs_emb = self.obs_encoder[view_name](imgs)
            imgs_emb = self.avgpool(imgs_emb)
            imgs_emb = imgs_emb.squeeze(-1).squeeze(-1)
            imgs_emb = imgs_emb.unsqueeze(1) # dummy patch dim
            imgs_emb = rearrange(imgs_emb, "(b t) p d -> b t p d", b=b)
            view_embs[view_name] = imgs_emb
        return view_embs


if __name__ == "__main__":
    ckpt = 'data/outputs/2025.03.26/04.49.31_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/240.ckpt'
    encoder = ResNetEncoder(ckpt, ['sideview', 'robot0_eye_in_hand'])
    bs = 2
    x_view_1 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x_view_2 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x = {'sideview': x_view_1, 'robot0_eye_in_hand': x_view_2}
    view_embs = encoder(x)
    for view_name, emb in view_embs.items():
        print(f"{view_name}: {emb.shape}")

