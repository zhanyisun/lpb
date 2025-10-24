import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

class VisualDynamicsModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        train_encoder=True,
        train_predictor=False,
        view_names=['view1'],
        use_layernorm=True,
        language_encoder=None,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor 
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.view_names = view_names
        self.language_encoder = language_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.emb_dim = self.encoder.emb_dim * len(self.view_names) + (self.action_dim + self.proprio_dim)

        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")
        print(f'image_size: {image_size}')

        print("Model emb_dim: ", self.emb_dim)

        self.encoder_transform = lambda x: x

        self.emb_criterion = nn.MSELoss()

        if use_layernorm:
            self.per_view_norm = nn.ModuleDict({
                view_name: nn.LayerNorm(self.encoder.emb_dim, elementwise_affine=False)
                for view_name in view_names
            }).to("cuda")
            if len(view_names) > 1: 
                total_dim = self.encoder.emb_dim * len(view_names)
                self.fusion_norm = nn.LayerNorm(total_dim, elementwise_affine=False).to("cuda")

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, 1)
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        act_repeated = act_tiled.repeat(1, 1, 1, 1)
        if 'language' not in z_dct:
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        else:
            language_repeated = z_dct['language'].unsqueeze(1).repeat(1, proprio_repeated.shape[1], 1, 1)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated, language_repeated], dim=3
            )
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        visual_embs = self.encode_obs_visual(obs['visual'])
        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        res = {"visual": visual_embs, "proprio": proprio_emb}
        if 'language' in obs:
            res['language'] = obs['language']
        return res

    def encode_obs_visual(self, obs_visual):
        view_embs = self.encoder(obs_visual)
        for view_name in self.view_names:
            if hasattr(self, "per_view_norm"):
                view_embs[view_name] = self.per_view_norm[view_name](view_embs[view_name])
        visual_embs = torch.cat([view_embs[view_name] for view_name in self.view_names], dim=-1)
        if hasattr(self, "fusion_norm"):
            visual_embs = self.fusion_norm(visual_embs)
            
        return visual_embs

    def predict(self, z):
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z
  
    def compute_recon_loss(self, obs_pred, obs_tgt, criterion):
        loss = 0
        for view_name in self.view_names:
            loss += criterion(obs_pred[view_name], obs_tgt[view_name])
        return loss
    
    def separate_emb(self, z):
        z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                        z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                        z[..., -self.action_dim:]
        # remove tiled dimensions
        z_proprio = z_proprio[:, :, 0, : self.proprio_dim]
        z_act = z_act[:, :, 0, : self.action_dim]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        loss = 0.0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]
        if 'language' in obs:
            z_tgt = z_tgt[..., :-32]    

        z_pred = self.predict(z_src)

        z_visual_loss = self.emb_criterion(
            z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
            z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
        )
        z_proprio_loss = self.emb_criterion(
            z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
            z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
        )
        z_loss = self.emb_criterion(
            z_pred[:, :, :, :-self.action_dim], 
            z_tgt[:, :, :, :-self.action_dim].detach()
        )

        loss = loss + z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss

        loss_components["loss"] = loss
        return loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)

        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
        act_repeated = act_tiled.repeat(1, 1, 1, 1)
        
        z_updated = torch.cat([
            z[..., :-self.action_dim],
            act_repeated
        ], dim=-1)

        return z_updated
