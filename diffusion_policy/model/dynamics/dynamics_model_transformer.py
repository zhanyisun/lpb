from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Identity

import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim, pretrained=True, backbone_frozen=False, project_to_latent=False):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.project_to_latent = project_to_latent
        if backbone_frozen:
            print('Freezing ResNet backbone')
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = Identity()
        if self.project_to_latent:
            self.projector = nn.Sequential(
                nn.Linear(512, 256),  # First layer to reduce dimensionality
                nn.ReLU(),            # Activation
                nn.BatchNorm1d(256),  # Batch normalization for stability
                nn.Linear(256, latent_dim)  # Project to latent_dim
            )

        # self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal Initialization for Conv layers with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BatchNorm layers initialized to weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if isinstance(m, nn.Sequential):
                    continue  # Skip if part of Sequential (handled separately)
                # Kaiming Normal for layers with ReLU, else Xavier Normal
                if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize projector layers separately if using Sequential
        if self.project_to_latent:
            for m in self.projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, img):
        """
        Args:
            img: Tensor of shape (batch_size, 3, H, W)
        Returns:
            latent: Tensor of shape (batch_size, latent_dim)
        """
        features = self.resnet(img)  # (batch_size, 512)
        # latent = self.projector(features)  # (batch_size, latent_dim)
        if self.project_to_latent:
            features = self.projector(features)
        return features
    

class ImageDecoder(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_height=96, 
                 image_width=96, 
                 image_channels=3, 
                 dropout_rate=0.0):
        super(ImageDecoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        
        # Number of times we upsample is fixed to 4, so we reduce the spatial resolution
        # by factor of 2^4 = 16 in the fully-connected layers:
        self.init_h = image_height // 16
        self.init_w = image_width // 16
        
        # Fully-connected block
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512 * self.init_h * self.init_w),
            nn.ReLU()
        )
        
        # Deconvolution block (4 layers, each doubling spatial size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # self._initialize_weights()

    def _initialize_weights(self):
        # Initialize fully connected layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize deconvolutional layers
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Tanh):
                # Tanh has no learnable weights but we keep this for clarity
                pass

    def forward(self, z):
        """
        Forward pass of the decoder.
        
        Args:
            z (Tensor): A latent vector of shape (batch_size, latent_dim)
        
        Returns:
            reconstructed_img (Tensor): The reconstructed image of shape 
                                        (batch_size, image_channels, image_height, image_width)
        """
        x = self.fc(z)  # -> (batch_size, 512 * init_h * init_w)
        x = x.view(-1, 512, self.init_h, self.init_w)  # (batch_size, 512, init_h, init_w)
        reconstructed_img = self.deconv(x)  # (batch_size, image_channels, image_height, image_width)
        return reconstructed_img
    

class ImageDecoder84(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_height=84, 
                 image_width=84, 
                 image_channels=3, 
                 dropout_rate=0.0):
        super(ImageDecoder84, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        
        # Calculate initial spatial dimensions
        self.init_h = image_height // 4  # 84 // 4 = 21
        self.init_w = image_width // 4    # 84 // 4 = 21
        
        # Fully-connected block
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512 * self.init_h * self.init_w),
            nn.ReLU()
        )
        
        # Deconvolution block (2 layers, each doubling spatial size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 21 -> 42
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 42 -> 84
            nn.ReLU(),
            nn.ConvTranspose2d(128, image_channels, kernel_size=3, stride=1, padding=1),  # 84 -> 84
            nn.Tanh()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize fully connected layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize deconvolutional layers
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Tanh):
                # Tanh has no learnable parameters
                pass

    def forward(self, z):
        """
        Forward pass of the decoder.
        
        Args:
            z (Tensor): A latent vector of shape (batch_size, latent_dim)
        
        Returns:
            reconstructed_img (Tensor): The reconstructed image of shape 
                                        (batch_size, image_channels, image_height, image_width)
        """
        x = self.fc(z)  # -> (batch_size, 512 * 21 * 21)
        x = x.view(-1, 512, self.init_h, self.init_w)  # (batch_size, 512, 21, 21)
        reconstructed_img = self.deconv(x)  # (batch_size, 3, 84, 84)
        return reconstructed_img
    

class TransformerDynamicsModelDecoderOnly(nn.Module):
    def __init__(self, 
            latent_dim: int,
            action_dim: int,
            horizon: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal: bool=False):
        super(TransformerDynamicsModelDecoderOnly, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal
        
        self.action_emb = nn.Linear(self.action_dim, n_emb)
        self.state_emb = nn.Linear(self.latent_dim, n_emb)
        self.z_pos_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True, 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, latent_dim)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        ignore_types = (nn.Dropout,  
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelDecoderOnly):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.z_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("z_pos_emb")
        no_decay.add("action_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    
    def forward(self, z_t, actions):
        """
        Args:
            z_t: (bs, latent_dim)
            actions: (bs, horizon, action_dim)
        Returns:
            \hat{z_future}: (bs, horizon, latent_dim)
        """
        # bs = z_t.size(0)
        # device = z_t.device
        z_emb = self.state_emb(z_t).unsqueeze(1)  # (bs, 1, n_emb)
        action_emb = self.action_emb(actions) # (bs, horizon, n_emb)

        x = self.drop(z_emb + self.z_pos_emb)
        memory = x

        token_embeddings = action_emb
        x = self.drop(token_embeddings + self.action_pos_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
        )
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


class TransformerDynamicsModelEncoderDecoder(nn.Module):
    def __init__(self, 
            latent_dim: int,
            action_dim: int,
            horizon: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal: bool=False):
        super(TransformerDynamicsModelEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal
        self.n_emb = n_emb
        # Embeddings
        self.action_emb = nn.Linear(self.action_dim, n_emb)
        self.state_emb = nn.Linear(self.latent_dim, n_emb)
        # Positional Embeddings
        self.z_pos_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.future_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        # Define a learnable decoder start token
        self.decoder_start_token = nn.Parameter(torch.zeros(1, 1, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True, 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, latent_dim)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        ignore_types = (nn.Dropout,  
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelEncoderDecoder):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.z_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.future_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.decoder_start_token, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("z_pos_emb")
        no_decay.add("action_pos_emb")
        no_decay.add("future_pos_emb")
        no_decay.add("decoder_start_token")
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def gen_tgt_mask(self, tgt_size, device):
        return nn.Transformer.generate_square_subsequent_mask(tgt_size).to(device)

    def gen_memory_mask(self, tgt_size, memory_size, device):
        tgt_indices = torch.arange(tgt_size, device=device).unsqueeze(1)  # (T, 1)
        memory_indices = torch.arange(memory_size, device=device).unsqueeze(0)  # (1, S)
        mask = memory_indices > (tgt_indices + 1)
        return mask  # (T, S)
    
    def forward(self, z_t, actions):
        """
        Args:
            z_t: (bs, latent_dim)
            actions: (bs, horizon, action_dim)
        Returns:
            \hat{z_future}: (bs, horizon, latent_dim)
        """
        bs = z_t.size(0)
        # device = z_t.device
        # Encode the current state
        z_emb = self.state_emb(z_t).unsqueeze(1)  # (bs, 1, n_emb)
        z_emb = self.drop(z_emb + self.z_pos_emb)

        # Encode the actions
        a_emb = self.action_emb(actions)  # (bs, horizon, n_emb)
        a_emb = self.drop(a_emb + self.action_pos_emb)

        encoder_input = torch.cat([z_emb, a_emb], dim=1)  # (batch_size, 1 + horizon, n_emb)
        encoder_output = self.encoder(encoder_input) # (batch_size, 1 + horizon, n_emb)

        # # NOTE: this is likely a bug
        # decoder_input = self.future_pos_emb.repeat(bs, 1, 1)  # (batch_size, horizon, n_emb)
        # decoder_input = self.drop(decoder_input)

        # Prepare decoder input using the decoder start token
        decoder_input = self.decoder_start_token.expand(bs, self.horizon, self.n_emb)
        decoder_input = self.drop(decoder_input + self.future_pos_emb)

        tgt_mask, memory_mask = None, None
        if self.causal:
            tgt_mask = self.gen_tgt_mask(self.horizon, z_t.device)
            memory_mask = self.gen_memory_mask(decoder_input.shape[1], encoder_output.shape[1], z_t.device)

        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


class TransformerDynamicsModelInterleaveEncoderDecoder(nn.Module):
    def __init__(self, 
            latent_dim: int,
            action_dim: int,
            horizon: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal: bool=False):
        super(TransformerDynamicsModelInterleaveEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal

        # Embeddings
        self.action_emb = nn.Linear(self.action_dim, n_emb)
        self.state_emb = nn.Linear(self.latent_dim, n_emb)

        # Positional Embeddings
        max_seq_length = 1 + 2 * horizon  # For interleaved sequence
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_length, n_emb))
        self.future_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4,
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer,
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, latent_dim)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelInterleaveEncoderDecoder):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.future_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        # The implementation remains the same, adjust if necessary
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Special case for position embeddings
        no_decay.add("pos_emb")
        no_decay.add("future_pos_emb")

        # Validate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not separated!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def gen_tgt_mask(self, tgt_size, device):
        return nn.Transformer.generate_square_subsequent_mask(tgt_size).to(device)

    def forward(self, z_t, actions):
        """
        Args:
            z_t: (batch_size, latent_dim)
            actions: (batch_size, horizon, action_dim)
        Returns:
            x: (batch_size, horizon, latent_dim)
        """
        batch_size = z_t.size(0)
        device = z_t.device

        # Encode the current state
        z_emb = self.state_emb(z_t).unsqueeze(1)  # Shape: (batch_size, 1, n_emb)

        # Encode the actions
        a_emb = self.action_emb(actions)  # Shape: (batch_size, horizon, n_emb)

        # Create placeholder embeddings for future states
        placeholder_z_emb = torch.zeros_like(z_emb).expand(-1, self.horizon, -1)  # (batch_size, horizon, n_emb)

        # Build the input sequence by interleaving state and action embeddings
        input_sequence = [z_emb]
        for i in range(self.horizon):
            input_sequence.append(a_emb[:, i:i+1, :])          # Action at time t+i
            input_sequence.append(placeholder_z_emb[:, i:i+1, :])  # Placeholder for state at time t+i+1

        # Concatenate the input sequence
        encoder_input = torch.cat(input_sequence, dim=1)  # Shape: (batch_size, 1 + 2 * horizon, n_emb)

        # Apply positional embeddings and dropout
        seq_length = encoder_input.size(1)
        pos_emb = self.pos_emb[:, :seq_length, :]
        encoder_input = self.drop(encoder_input + pos_emb)

        # Generate encoder mask
        if self.causal:
            encoder_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()
        else:
            encoder_mask = None

        # Encoder forward pass
        encoder_output = self.encoder(encoder_input, mask=encoder_mask)  # Shape: (batch_size, seq_length, n_emb)

        # Prepare decoder input (placeholder embeddings with positional encoding)
        decoder_input = placeholder_z_emb + self.future_pos_emb  # Shape: (batch_size, horizon, n_emb)
        decoder_input = self.drop(decoder_input)

        # Generate target mask for decoder
        if self.causal:
            tgt_mask = self.gen_tgt_mask(self.horizon, device)
        else:
            tgt_mask = None

        # Decoder forward pass
        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=None,
        )  # Shape: (batch_size, horizon, n_emb)

        x = self.ln_f(x)
        x = self.head(x)  # Shape: (batch_size, horizon, latent_dim)

        return x


class TransformerDynamicsModelActionPredEncoderDecoder(nn.Module):
    def __init__(self, 
            latent_dim: int,
            action_dim: int,
            horizon: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal: bool=False):
        super(TransformerDynamicsModelActionPredEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal
        self.n_emb = n_emb

        # Embeddings
        self.action_emb = nn.Linear(self.action_dim, n_emb)
        self.state_emb = nn.Linear(self.latent_dim, n_emb)

        # Positional Embeddings
        self.z_pos_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.future_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        # Define a learnable decoder start token
        self.decoder_start_token = nn.Parameter(torch.zeros(1, 1, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True, 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        self.ln_f = nn.LayerNorm(n_emb)
        # Adjust the output layer to output both latent states and actions
        self.head = nn.Linear(n_emb, latent_dim + action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,  
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelActionPredEncoderDecoder):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.z_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.future_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.decoder_start_token, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        # ... [The implementation remains the same] ...
        # No changes needed in this method
        # Ensure that any new parameters are appropriately included in the optimizer

        # [Existing code]
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("z_pos_emb")
        no_decay.add("action_pos_emb")
        no_decay.add("future_pos_emb")
        no_decay.add("decoder_start_token")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def gen_tgt_mask(self, tgt_size, device):
        return nn.Transformer.generate_square_subsequent_mask(tgt_size).to(device)

    def gen_memory_mask(self, tgt_size, memory_size, device):
        tgt_indices = torch.arange(tgt_size, device=device).unsqueeze(1)  # (T, 1)
        memory_indices = torch.arange(memory_size, device=device).unsqueeze(0)  # (1, S)
        mask = memory_indices > (tgt_indices + 1)
        return mask  # (T, S)
    
    def forward(self, z_t, actions):
        """
        Args:
            z_t: (batch_size, latent_dim)
            actions: (batch_size, horizon, action_dim)
        Returns:
            z_hat_future: (batch_size, horizon, latent_dim)
            a_hat_future: (batch_size, horizon, action_dim)
        """
        batch_size = z_t.size(0)
        device = z_t.device
        # Encode the current state
        z_emb = self.state_emb(z_t).unsqueeze(1)  # (batch_size, 1, n_emb)
        z_emb = self.drop(z_emb + self.z_pos_emb)

        # Encode the actions
        a_emb = self.action_emb(actions)  # (batch_size, horizon, n_emb)
        a_emb = self.drop(a_emb + self.action_pos_emb)

        # Concatenate the state and action embeddings
        encoder_input = torch.cat([z_emb, a_emb], dim=1)  # (batch_size, 1 + horizon, n_emb)
        encoder_output = self.encoder(encoder_input)  # (batch_size, 1 + horizon, n_emb)

        # Prepare decoder input using the decoder start token
        decoder_input = self.decoder_start_token.repeat(batch_size, self.horizon, 1)
        decoder_input = self.drop(decoder_input + self.future_pos_emb)

        # Generate masks if causal
        tgt_mask, memory_mask = None, None
        if self.causal:
            tgt_mask = self.gen_tgt_mask(self.horizon, device)
            memory_mask = self.gen_memory_mask(decoder_input.shape[1], encoder_output.shape[1], device)

        # Decoder forward pass
        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )  # (batch_size, horizon, n_emb)

        x = self.ln_f(x)
        output = self.head(x)  # (batch_size, horizon, latent_dim + action_dim)

        # Split the output into predicted latent states and actions
        z_hat_future = output[..., :self.latent_dim]     # (batch_size, horizon, latent_dim)
        a_hat_future = output[..., self.latent_dim:]     # (batch_size, horizon, action_dim)

        return z_hat_future, a_hat_future

     
class TransformerDynamicsModelEncoderDecoderHistory(nn.Module):
    def __init__(self, 
            latent_dim: int,
            action_dim: int,
            horizon: int,
            n_layer: int,
            n_head: int,
            n_emb: int,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal: bool=False):
        super(TransformerDynamicsModelEncoderDecoderHistory, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal
        # Embeddings
        self.action_emb = nn.Linear(self.action_dim, n_emb)
        self.state_emb = nn.Linear(self.latent_dim, n_emb)
        # Positional Embeddings
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.history_pos_emb = nn.Parameter(torch.zeros(1, horizon * 2 + 1, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True, 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, latent_dim)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        ignore_types = (nn.Dropout,  
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelEncoderDecoderHistory):
            torch.nn.init.normal_(module.action_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.history_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("action_pos_emb")
        no_decay.add("history_pos_emb")
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    
    def gen_tgt_mask(self, tgt_size, device):
        return nn.Transformer.generate_square_subsequent_mask(tgt_size).to(device)

    def forward(self, history_z, history_a, future_a):
        """
        Args:
            z_t: (bs, latent_dim)
            actions: (bs, horizon, action_dim)
        Returns:
            \hat{z_future}: (bs, horizon, latent_dim)
        """
        bs = history_z.size(0)
        # device = z_t.device
        # Encode history
        history_z_emb = self.state_emb(history_z)
        history_a_emb = self.action_emb(history_a)
        history_emb = torch.cat([history_z_emb, history_a_emb], dim=1)  # (bs, 2 * horizon + 1, n_emb)
        
        history_emb = self.drop(history_emb + self.history_pos_emb)

        # Encode the actions
        a_emb = self.action_emb(future_a)  # (bs, horizon, n_emb)
        a_emb = self.drop(a_emb + self.action_pos_emb)

        encoder_input = history_emb
        encoder_output = self.encoder(encoder_input) # (batch_size, 1 + horizon, n_emb)

        decoder_input = a_emb
        
        tgt_mask = None
        if self.causal:
            tgt_mask = self.gen_tgt_mask(decoder_input.shape[1], history_z.device)

        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=None,
        )
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


class TransformerDynamicsModelHistoryEncoderOnly(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        horizon: int,
        n_layer: int,
        n_head: int,
        n_emb: int,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal: bool = False,
    ):
        super(TransformerDynamicsModelHistoryEncoderOnly, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.causal = causal
        self.n_emb = n_emb
        # Embeddings
        self.state_emb = nn.Linear(self.latent_dim, n_emb)
        self.action_emb = nn.Linear(self.action_dim, n_emb)

        # Positional Embeddings
        max_seq_length = 4 * self.horizon + 1  # For interleaved sequence
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_length, n_emb))

        # Placeholder Embedding for future states
        self.placeholder_emb = nn.Parameter(torch.zeros(1, 1, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer,
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, latent_dim)  # Output head to predict future states
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,  
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerDynamicsModelHistoryEncoderOnly):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.placeholder_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("placeholder_emb")
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def gen_tgt_mask(self, tgt_size, device):
        return nn.Transformer.generate_square_subsequent_mask(tgt_size).to(device)

    def gen_memory_mask(self, tgt_size, memory_size, device):
        tgt_indices = torch.arange(tgt_size, device=device).unsqueeze(1)  # (T, 1)
        memory_indices = torch.arange(memory_size, device=device).unsqueeze(0)  # (1, S)
        mask = memory_indices > (tgt_indices + 1)
        return mask  # (T, S)

    def forward(self, history_z, history_a, future_a):
        """
        Args:
            history_z: (batch_size, horizon + 1, latent_dim)
            history_a: (batch_size, horizon, action_dim)
            future_a:  (batch_size, horizon, action_dim)
        Returns:
            x: (batch_size, horizon, latent_dim)
        """
        bs = history_z.size(0)
        device = history_z.device

        # Encode history states and actions
        history_z_emb = self.state_emb(history_z)
        history_a_emb = self.action_emb(history_a)

        # Interleave history states and actions
        input_sequence = []
        for i in range(self.horizon):
            input_sequence.append(history_z_emb[:, i:i+1, :])  # z_{t-h+i}
            input_sequence.append(history_a_emb[:, i:i+1, :])  # a_{t-h+i}
        input_sequence.append(history_z_emb[:, self.horizon:self.horizon+1, :])  # z_t

        # Encode future actions
        future_a_emb = self.action_emb(future_a)
        placeholder_z_emb = self.placeholder_emb.expand(bs, self.horizon, self.n_emb)

        check_future_z_emb_indices = []
        # Interleave future actions and placeholders
        for i in range(self.horizon):
            input_sequence.append(future_a_emb[:, i:i+1, :])        # a_{t+i}
            input_sequence.append(placeholder_z_emb[:, i:i+1, :])    # Placeholder for z_{t+i+1}
            check_future_z_emb_indices.append(len(input_sequence) - 1)
        # TODO: check future_state_indices
        # Concatenate the input sequence
        encoder_input = torch.cat(input_sequence, dim=1)  # Shape: (batch_size, seq_length, n_emb)

        # Apply positional embeddings and dropout
        seq_length = encoder_input.size(1)
        pos_emb = self.pos_emb[:, :seq_length, :]
        encoder_input = self.drop(encoder_input + pos_emb)

        # Generate encoder mask if causal
        if self.causal:
            encoder_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()
        else:
            encoder_mask = None

        # Encoder forward pass
        encoder_output = self.encoder(encoder_input, mask=encoder_mask)

        # Extract future state embeddings from encoder output
        future_state_indices = [2 * self.horizon + 2 + 2 * i for i in range(self.horizon)]
        # assert (check_future_z_emb_indices == future_state_indices)
        future_state_embeddings = encoder_output[:, future_state_indices, :]  # Shape: (batch_size, horizon, n_emb)

        # Apply final layer normalization and output head
        x = self.ln_f(future_state_embeddings)
        x = self.head(x)  # Shape: (batch_size, horizon, latent_dim)

        return x



class FullModel(nn.Module):
    def __init__(self, 
                latent_dim: int,
                action_dim: int,
                horizon: int,
                n_layer: int,
                n_head: int,
                n_emb: int,
                p_drop_emb: float = 0.1,
                p_drop_attn: float = 0.1,
                image_height: int = 96,
                image_width: int = 96,
                causal: bool=False,
                pretrained_encoder=False, 
                backbone_frozen=False,
                project_to_latent=False,
                decode=False,
                decode_predicted=False,
                decode_gt=False,
                decode_both=True,
                interleave=False,
                pred_action=False):    
        super(FullModel, self).__init__()
        self.pred_action = pred_action
        self.obs_encoder = ImageEncoder(latent_dim, pretrained=pretrained_encoder, backbone_frozen=backbone_frozen, project_to_latent=project_to_latent)
        if pred_action:
            print('Using action prediction model')
            self.dynamics_model = TransformerDynamicsModelActionPredEncoderDecoder(latent_dim, action_dim, horizon, n_layer, n_head, n_emb, p_drop_emb, p_drop_attn, causal)
        elif interleave:
            print('Using interleave model')
            self.dynamics_model = TransformerDynamicsModelInterleaveEncoderDecoder(latent_dim, action_dim, horizon, n_layer, n_head, n_emb, p_drop_emb, p_drop_attn, causal)
        else:
            print('basic encoder-decoder model')
            self.dynamics_model = TransformerDynamicsModelEncoderDecoder(latent_dim, action_dim, horizon, n_layer, n_head, n_emb, p_drop_emb, p_drop_attn, causal)
        self.horizon = horizon
        self.latent_dim = latent_dim

        self.decode = decode
        self.decode_predicted = decode_predicted
        self.decode_gt = decode_gt
        self.decode_both = decode_both
        if self.decode:
            if image_height >= 96 and image_width >= 96:
                self.obs_decoder = ImageDecoder(latent_dim=latent_dim, image_height=image_height, image_width=image_width) 
            else:
                self.obs_decoder = ImageDecoder84(latent_dim=latent_dim, image_height=image_height, image_width=image_width)

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.dynamics_model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        if self.decode:
            optim_groups.append({
                "params": self.obs_decoder.parameters(),
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, obs, actions, obs_future):
        h, w = obs.shape[-2], obs.shape[-1]
        bs = obs.size(0)
        z_t = self.obs_encoder(obs)
        a_hat_future = None
        if self.pred_action:
            z_hat_future, a_hat_future = self.dynamics_model(z_t, actions)
        else:
            z_hat_future = self.dynamics_model(z_t, actions)
        recon_s_future = None
        if self.decode:
            if self.decode_predicted or self.decode_both:
                recon_s_future_from_predicted = self.obs_decoder(z_hat_future.reshape(-1, self.latent_dim))
                recon_s_future_from_predicted = recon_s_future_from_predicted.reshape(bs, self.horizon, 3, h, w)
                recon_s_future = {'from_predicted': recon_s_future_from_predicted}
            if self.decode_gt or self.decode_both:
                z_future = self.obs_encoder(obs_future.reshape(-1, 3, h, w)).reshape(-1, self.latent_dim)
                recon_s_future_from_gt = self.obs_decoder(z_future)
                recon_s_future_from_gt = recon_s_future_from_gt.reshape(bs, self.horizon, 3, h, w)
                recon_s_future = {'from_gt': recon_s_future_from_gt}
            if self.decode_both:
                recon_s_future = {'from_predicted': recon_s_future_from_predicted, 'from_gt': recon_s_future_from_gt}
        return z_t, z_hat_future, recon_s_future, a_hat_future

    def z_future(self, obs_future):
        h, w = obs_future.shape[-2], obs_future.shape[-1]
        with torch.no_grad():
            z_future = self.obs_encoder(obs_future.reshape(-1, 3, h, w)).reshape(-1, self.horizon, self.latent_dim)
        return z_future
    
    def encode_obs(self, obs):
        return self.obs_encoder(obs)


class FullModelHistory(nn.Module):
    def __init__(self, 
                latent_dim: int,
                action_dim: int,
                horizon: int,
                n_layer: int,
                n_head: int,
                n_emb: int,
                p_drop_emb: float = 0.1,
                p_drop_attn: float = 0.1,
                causal: bool=False,
                pretrained_encoder=True, 
                backbone_frozen=True,
                decode=False,
                history_encoder_only=False):    
        super(FullModelHistory, self).__init__()
        self.obs_encoder = ImageEncoder(latent_dim, pretrained=pretrained_encoder, backbone_frozen=backbone_frozen)
        if not history_encoder_only:
            self.dynamics_model = TransformerDynamicsModelEncoderDecoderHistory(latent_dim, action_dim, horizon, n_layer, n_head, n_emb, p_drop_emb, p_drop_attn, causal)
        else:
            self.dynamics_model = TransformerDynamicsModelHistoryEncoderOnly(latent_dim, action_dim, horizon, n_layer, n_head, n_emb, p_drop_emb, p_drop_attn, causal)

        self.horizon = horizon
        self.latent_dim = latent_dim

        self.decode = decode
        if self.decode:
            self.obs_decoder = ImageDecoder(latent_dim=latent_dim, image_height=208, image_width=320) 

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.dynamics_model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        if self.decode:
            optim_groups.append({
                "params": self.obs_decoder.parameters(),
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, history_obs, history_actions, future_actions):
        bs = history_obs.size(0)
        h, w = history_obs.shape[-2], history_obs.shape[-1]

        z_history = self.obs_encoder(history_obs.reshape(-1, 3, h, w)).reshape(-1, self.horizon + 1, self.latent_dim)
        z_hat_future = self.dynamics_model(z_history, history_actions, future_actions)
        recon_s_future = None
        if self.decode:
            recon_s_future = self.obs_decoder(z_hat_future.reshape(-1, self.latent_dim)).reshape(bs, self.horizon, 3, h, w)
        return z_history, z_hat_future, recon_s_future

    def z_future(self, obs_future):
        with torch.no_grad():
            z_future = self.obs_encoder(obs_future.reshape(-1, 3, h, w)).reshape(-1, self.horizon, self.latent_dim)
        return z_future
    
    def encode_obs(self, obs):
        return self.obs_encoder(obs)


def test():
    model = FullModel(latent_dim=1024, action_dim=10, image_height=84, image_width=84, horizon=8, n_layer=6, n_head=8, n_emb=256, decode=True, pred_action=True)
    opt = model.get_optimizer(1e-3, 1e-3, 1e-4, (0.9, 0.95))
    # history_obs = torch.rand(32, 3, 11, 208, 320)
    # history_actions = torch.rand(32, 10, 4)
    actions = torch.rand(32, 8, 10)
    obs = torch.rand(32, 3, 84, 84)
    z_t, z_hat_future, recon, _ = model(obs, actions)


if __name__ == "__main__":
    test()