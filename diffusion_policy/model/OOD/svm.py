import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM

class SVMOODModule(nn.Module):
    def __init__(self, svm_model):
        super(SVMOODModule, self).__init__()
        self.svm = svm_model

    def forward(self, z):
        z_np = z.cpu().detach().numpy()  # Convert to numpy for scikit-learn
        decision_scores = self.svm.decision_function(z_np)
        decision_scores = torch.tensor(decision_scores, device=z.device, dtype=torch.float32)
        return -decision_scores 
    

def train_ocsvm(train_latents, nu=0.1, gamma='scale'):
    oc_svm = OneClassSVM(nu=nu, gamma=gamma)
    oc_svm.fit(train_latents.detach().cpu().numpy())
    return oc_svm