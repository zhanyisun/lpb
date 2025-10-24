import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(RealNVP, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )
        self.translation_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )
        self.scale_net[-1].weight.data.zero_()
        self.scale_net[-1].bias.data.zero_()
        self.translation_net[-1].weight.data.zero_()
        self.translation_net[-1].bias.data.zero_()

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if reverse:
            s = self.scale_net(x1)
            t = self.translation_net(x1)
            x2 = (x2 - t) * torch.exp(-s)
        else:
            s = self.scale_net(x1)
            t = self.translation_net(x1)
            x2 = x2 * torch.exp(s) + t

        z = torch.cat([x1, x2], dim=1)
        log_det_jacobian = s.sum(dim=1)
        if reverse:
            log_det_jacobian *= -1
        return z, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_flows=4):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([RealNVP(latent_dim, hidden_dim) for _ in range(num_flows)])
        self.base_dist = None  # Placeholder for the base distribution

    def forward(self, x):
        log_det_jacobian = 0
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian

    def inverse(self, z):
        x = z
        log_det_jacobian = 0
        for flow in reversed(self.flows):
            x, log_det = flow(x, reverse=True)
            log_det_jacobian += log_det
        return x, log_det_jacobian

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        if self.base_dist is None:
            # Initialize the base distribution on the same device as the latent vectors
            device = x.device
            self.base_dist = D.MultivariateNormal(torch.zeros(x.shape[1], device=device), torch.eye(x.shape[1], device=device))
        log_prob = self.base_dist.log_prob(z) + log_det_jacobian
        return log_prob

class NormalizingFlowOODModule(nn.Module):
    def __init__(self, flow_model):
        super(NormalizingFlowOODModule, self).__init__()
        self.flow_model = flow_model

    def forward(self, z):
        log_prob = self.flow_model.log_prob(z)
        return -log_prob

# Function to train the normalizing flow
def train_normalizing_flow(train_latents, latent_dim, hidden_dim, num_flows=4, epochs=100, lr=1e-3):
    flow_model = NormalizingFlow(latent_dim, hidden_dim, num_flows).cuda()
    optimizer = optim.Adam(flow_model.parameters(), lr=lr)

    for epoch in range(epochs):
        flow_model.train()
        optimizer.zero_grad()
        log_prob = flow_model.log_prob(train_latents)
        loss = -log_prob.mean()  # Maximize log-prob (equivalent to minimizing negative log-prob)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    return flow_model

# Example dataset loader (dummy implementation)
def get_train_loader(batch_size=32):
    # Dummy dataset: Latent vectors sampled from a normal distribution
    train_latents = torch.randn(1000, 32).cuda()  # 1000 latent vectors, each of size 32
    train_loader = DataLoader(train_latents, batch_size=batch_size, shuffle=True)
    return train_loader

# Main code
if __name__ == "__main__":
    # Step 1: Prepare the training data (dummy latent vectors)
    train_loader = get_train_loader()
    all_train_latents = []
    for batch in train_loader:
        all_train_latents.append(batch)
    all_train_latents = torch.cat(all_train_latents, dim=0)  # Shape: [1000, 32]

    # Step 2: Train the normalizing flow on the latent vectors
    latent_dim = 32
    hidden_dim = 64
    flow_model = train_normalizing_flow(all_train_latents, latent_dim, hidden_dim)

    # Step 3: Initialize the OOD module
    ood_module = NormalizingFlowOODModule(flow_model).cuda()

    # Step 4: Simulate an optimization step to minimize the OOD score
    batch_size = 32
    num_steps = 8
    action_dim = 2

    # Dummy initial state and action tensor with requires_grad=True
    initial_state = torch.randn(batch_size, 32, requires_grad=False, device='cuda')  # Latent vector
    action = torch.randn(batch_size, num_steps, action_dim, requires_grad=True, device='cuda')

    optimizer = optim.Adam([action], lr=1e-4)

    num_iters = 1
    for _ in range(num_iters):
        optimizer.zero_grad()
        total_ood_score = 0

        # Initialize the state with the initial state
        current_state = initial_state.clone()

        predicted_states = []
        for i in range(num_steps):
            current_action = action[:, i, :]

            # Predict the next state using the dynamics model (assume dynamics_model is defined)
            current_state = dynamics_model.latent_transition(current_state, current_action)
            predicted_states.append(current_state)

            # Compute the OOD score for the current state and accumulate it
            ood_score = ood_module(current_state)
            total_ood_score += ood_score.sum()

        # Backpropagate the total loss for all steps at once
        total_ood_score.backward()

        # Update the actions to minimize the OOD score
        optimizer.step()

    print("Optimization step complete.")
