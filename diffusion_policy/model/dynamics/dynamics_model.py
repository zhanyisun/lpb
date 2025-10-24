import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb


class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError("The encode method needs to be implemented by subclasses.")
    
    def latent_transition(self, z_t, action):
        # Concatenate the image latent, state, and action for prediction
        z_t_with_action = torch.cat([z_t, action], dim=1)
        z_t1_pred = self.dynamics_predictor(z_t_with_action)
        return z_t1_pred

    def init_weights(self):
        # Initialize weights using Xavier initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class StateEncodeBasedDynamicsModel(DynamicsModel):
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=32):
        super(StateEncodeBasedDynamicsModel, self).__init__()
        # Dynamics predictor MLP
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),  # input: state + action
            # nn.BatchNorm1d(hidden_dim), # works better without batchnorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, state, action):
        # Concatenate state and action
        z_t = self.encode(state)
        input_with_action = torch.cat([z_t, action], dim=1)
        # Predict the next state
        state_next_pred = self.dynamics_predictor(input_with_action)
        return z_t, state_next_pred
    
    def encode(self, state):
        return self.state_encoder(state)

class StateEncodeBasedDynamicsModelOptimize(DynamicsModel):
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=32, dropout_prob=0.5):
        super(StateEncodeBasedDynamicsModelOptimize, self).__init__()
        # State Encoder: Linear layer with BatchNorm, ReLU, and Dropout
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)  # Add dropout
        )
        
        # Dynamics Predictor MLP: Predicts next state based on state + action
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),  # input: encoded state + action
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Add dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2)  # Add batch normalization for final output layer
        )

    def forward(self, state, action):
        # Encode the state
        z_t = self.encode(state)
        
        # Concatenate encoded state and action
        input_with_action = torch.cat([z_t, action], dim=1)
        
        # Predict the next state
        state_next_pred = self.dynamics_predictor(input_with_action)
        
        return z_t, state_next_pred
    
    def encode(self, state):
        return self.state_encoder(state)
    
class StateBasedEncoderDecoderDynamicsModel(DynamicsModel):
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=32):
        super(StateBasedEncoderDecoderDynamicsModel, self).__init__()
        
        # Encoder: Encodes state into latent representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Dynamics predictor: Predicts next latent state based on current latent state and action
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),  # input: latent state + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: Decodes latent representation back into state space
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Reconstructs the original state
        )

    def forward(self, state, action):
        # Encode the current state into the latent space
        z_t = self.encode(state)
        
        # Predict the next latent state based on the action
        input_with_action = torch.cat([z_t, action], dim=1)
        z_t1_pred = self.dynamics_predictor(input_with_action)
        
        # Decode the current latent state back to the current state
        s_t_recon = self.decode(z_t)
        
        # Decode the predicted next latent state back to the next state
        s_t1_recon = self.decode(z_t1_pred)
        
        return z_t, z_t1_pred, s_t_recon, s_t1_recon

    def encode(self, state):
        return self.state_encoder(state)

    def decode(self, latent_state):
        return self.state_decoder(latent_state)


class StateBasedDynamicsModel(DynamicsModel):
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=32):
        super(StateBasedDynamicsModel, self).__init__()
        # Dynamics predictor MLP
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),  # input: state + action
            # nn.BatchNorm1d(hidden_dim), # works better without batchnorm
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        # Concatenate state and action
        z_t = self.encode(state)
        input_with_action = torch.cat([z_t, action], dim=1)
        # Predict the next state
        state_next_pred = self.dynamics_predictor(input_with_action)
        return z_t, state_next_pred
    
    def encode(self, state):
        return state


class ImageBasedDynamicsModel(DynamicsModel):
    def __init__(self, action_dim=2, latent_size=32):
        super(ImageBasedDynamicsModel, self).__init__()
        # Load a pretrained ResNet-18 model and modify the final layer
        resnet = models.resnet18(pretrained=False)
        self.img_encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, latent_size),
            nn.BatchNorm1d(latent_size),  # has to use batchnorm for image-based model
            nn.ReLU()
        )
        # Dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_size + action_dim, latent_size),  # input: latent_z_t + action
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )

    def forward(self, img, action):
        # Encode the image and state into a latent vector
        z_t = self.encode(img)
        # Predict next latent state given z_t and action
        z_t_with_action = torch.cat([z_t, action], dim=1)
        z_t1_pred = self.dynamics_predictor(z_t_with_action)
        return z_t, z_t1_pred

    def encode(self, img):
        img_latent = self.img_encoder(img)
        return img_latent
    

class HybridDynamicsModel(DynamicsModel):
    def __init__(self, state_dim=5, action_dim=2, latent_size=32):
        super(HybridDynamicsModel, self).__init__()
        # Load a pretrained ResNet-18 model and modify the final layer for image encoding
        resnet = models.resnet18(pretrained=False)
        self.img_encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU()
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_size // 2),
            # nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(),
            # nn.Linear(latent_size // 2, latent_size // 2),
            # nn.BatchNorm1d(latent_size // 2),
            # nn.ReLU()
        )


        self.latent_transform = nn.Sequential(
            nn.Linear(latent_size + latent_size // 2, latent_size),
            # nn.BatchNorm1d(latent_size),
            nn.ReLU()
        )

        # Dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_size + action_dim, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )

    def forward(self, img, state, action):
        # Encode the image into a latent vector
        z_t = self.encode(img, state)
        z_t_with_action = torch.cat([z_t, action], dim=1)
        z_t1_pred = self.dynamics_predictor(z_t_with_action)
        return z_t, z_t1_pred

    def encode(self, img, state):
        img_latent = self.img_encoder(img)
        state_latent = self.state_encoder(state)
        combined_latent = torch.cat([img_latent, state_latent], dim=1)
        # Transform to match latent_size
        return self.latent_transform(combined_latent)


class CVAE(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, latent_dim=4, detach_z_t1=False):
        super(CVAE, self).__init__()
        
        # Encoder: Encodes the state s_t into a latent vector z_t
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),  # Reduced hidden units
            nn.ReLU(),
            nn.Linear(32, 16),  # Reduced hidden units
            nn.ReLU(),
            nn.Linear(16, 2 * latent_dim // 2)  # Output mean and log variance
        )
        
        # Decoder: Reconstructs the next state s_{t+1} from z_t and a_t (action)
        # NOTE: if reconstructing images, we should use a sigmoid as activation and use bce loss as reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim // 2 + action_dim, 16),  # Reduced hidden units
            nn.ReLU(),
            nn.Linear(16, 32),  # Reduced hidden units
            nn.ReLU(),
            nn.Linear(32, state_dim),  # Output next state s_{t+1}
            nn.Tanh()  # Ensure output is within [-1, 1]
        )
        
        # Dynamics MLP: Predicts the next latent state z_{t+1} from z_t and a_t
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(latent_dim // 2 + action_dim, 16),  # Reduced hidden units
            nn.ReLU(),
            nn.Linear(16, latent_dim // 2)  # Reduced hidden units
        )

        self.detach_z_t1 = detach_z_t1
    
    def encode(self, state):
        mean_logvar = self.encoder(state)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar) * 0.5
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, action):
        # Concatenate the latent variable z_t with action a_t
        z_action = torch.cat([z, action], dim=-1)
        return self.decoder(z_action)
    
    def forward(self, state, action, next_state):
        # Encode current state s_t
        mean, logvar = self.encode(state)
        z_t = self.reparameterize(mean, logvar)
        
        # Encode next state s_{t+1} without reparameterization
        next_mean, _ = self.encode(next_state)
        z_t1 = next_mean  # True z_{t+1} directly from the mean
        
        # Predict the next latent state from z_t and a_t
        z_t1_pred = self.dynamics_mlp(torch.cat([z_t, action], dim=-1))
        
        # Decode (reconstruct) the next state s_{t+1} from z_t and a_t
        recon_next_state = self.decode(z_t, action)
        
        if self.detach_z_t1:
            z_t1_without_grad = z_t1.detach()
        else:
            z_t1_without_grad = z_t1

        return recon_next_state, z_t1_pred, z_t1, z_t1_without_grad, z_t, mean, logvar
    
    def latent_transition(self, z_t, action):
        # Concatenate the image latent, state, and action for prediction
        z_t_with_action = torch.cat([z_t, action], dim=1)
        z_t1_pred = self.dynamics_mlp(z_t_with_action)
        return z_t1_pred