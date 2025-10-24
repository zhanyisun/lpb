import math
import os
from diffusion_policy.common.cv2_util import save_images_to_video
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.linalg as linalg
import torch.nn.functional as F

class KNN_torch(nn.Module):
    def __init__(self, train_latents, images, method='soft_knn', temp=0.1, cont_loss='triplet_loss'):
        super(KNN_torch, self).__init__()
        # self.encoder = encoder
        if len(train_latents.shape) == 3:
            train_latents = train_latents.reshape(train_latents.size(0), -1)
        self.n = train_latents.size(0)
        self.d = train_latents.size(1)
        self.images = images
        self.nn_list = []
        self.current_image_list = []

        self.method = method
        self.temp = temp
        self.cont_loss = cont_loss
        print('loss is ', self.cont_loss)

        if self.method == 'mahalanobis':
            self.covariance_matrix = torch.cov(self.train_latents.T)
            self.cov_inv = linalg.inv(self.covariance_matrix)
            self.mean_latent = torch.mean(self.train_latents, dim=0)

        if self.method == 'kde':
            self.bandwidth = self.silverman_bandwidth()

        self.train_latents = F.normalize(train_latents, p=2, dim=-1)
        self.image_timestep = 0

    def forward(self, latent_eval, current_img=None, margin=0.0, p=1.0):
        if len(latent_eval.shape) == 3:
            latent_eval = latent_eval.reshape(latent_eval.size(0), -1)
        # print('method is ', self.method)
        latent_eval = F.normalize(latent_eval, p=2, dim=-1)
        distances = torch.cdist(latent_eval, self.train_latents, p=2)  # (batch_size, num_train_samples)
        min_distances, idx = distances.min(dim=1)
        if self.method == 'soft_knn':
            # # weights = torch.softmax(-distances / self.temp, dim=1)
            # # ood_score = (weights * distances).sum(dim=1)
            # latent_eval = F.normalize(latent_eval, p=2, dim=-1)
            # distances = torch.cdist(latent_eval, self.train_latents, p=2)  # (batch_size, num_train_samples)
            # min_distances, idx = distances.min(dim=1)
            # # Apply softmax over negative distances
            # weights = F.softmax(-distances / self.temp, dim=1)  # (batch_size, num_train_samples)
            # ood_score = (weights * distances).sum(dim=1)  # (batch_size,)
            if 'contrastive' in self.cont_loss or \
                'hybrid' in self.cont_loss or \
                'triplet' in self.cont_loss:
                sorted_distances, _ = distances.sort(dim=1)  # (batch_size, num_train_samples)
                top_k = 1
                if top_k > 1:
                    sorted_distances = sorted_distances[:, top_k - 1:]  # Ignore closest `top_k - 1` neighbors
                weights = F.softmax(-sorted_distances / self.temp, dim=1)  # (batch_size, num_train_samples - top_k + 1)
                ood_score = (weights * sorted_distances).sum(dim=1)  # (batch_size,)
            elif 'info_nce' in self.cont_loss:
                latent_eval = F.normalize(latent_eval, p=2, dim=-1)
                similarities = torch.matmul(latent_eval, self.train_latents.T)  # (batch_size, num_train_samples)
                weights = F.softmax(similarities / self.temp, dim=1)  # (batch_size, num_train_samples)
                ood_score = -(weights * similarities).sum(dim=1)  # (batch_size,)


        # elif self.method == 'cosine':
        #     latent_eval = F.normalize(latent_eval, p=2, dim=-1)  # (batch_size, latent_dim)
        #     # Compute cosine similarity with training embeddings
        #     similarities = torch.matmul(latent_eval, self.train_latents.T)  # (batch_size, num_train_samples)
        #     # Apply softmax over training samples (temperature-scaled)
        #     weights = F.softmax(similarities / 0.07, dim=1)  # (batch_size, num_train_samples)
        #     # Compute OOD score as the weighted sum of negative similarities
        #     ood_score = -torch.sum(weights * similarities, dim=1)  # (batch_size,)
        # elif self.method == 'vanilla_nn':
        #     min_distances, idx = distances.min(dim=1)
        #     margin_loss = torch.relu(min_distances - margin)  # Only penalizes if min_distance > margin
        #     ood_score = margin_loss.mean()
        # elif self.method == 'mahalanobis':
        #     diff = latent_eval - self.mean_latent
        #     mahalanobis_dist = torch.sqrt(torch.mm(torch.mm(diff, self.cov_inv), diff.T).diag())
        #     ood_score = mahalanobis_dist.mean()
        # elif self.method == 'energy':
        #     # Compute the energy score as the negative log-sum-exp of distances
        #     latent_eval = torch.nn.functional.normalize(latent_eval, p=2, dim=1)
        #     train_latents_normalized = torch.nn.functional.normalize(self.train_latents, p=2, dim=1)
        #     distances = torch.cdist(latent_eval, train_latents_normalized)
        #     max_dist = distances.max(dim=1, keepdim=True)[0]
        #     energy_score = torch.logsumexp(-(distances - max_dist), dim=1).mean()
        #     ood_score = -energy_score
        # elif self.method == 'scaled_nn':
        #     scaled_loss = min_distances ** p  # Using p > 1 to amplify gradients when distances are large
        #     ood_score = scaled_loss.mean()
        # elif self.method == 'kde':
        #     # Compute the difference between z_pred and each z_i in z_demo
        #     diff = (self.train_latents - latent_eval) / self.bandwidth  # Shape: (N, D)
        #     # Compute the squared distances
        #     squared_dist = torch.sum(diff ** 2, dim=1)  # Shape: (N,)

        #     # Log to check if squared distances are becoming too large
        #     if torch.isnan(squared_dist).any() or torch.isinf(squared_dist).any():
        #         print(f"Squared distances contain NaN or Inf values: {squared_dist}")
            
        #     # Compute the kernel values using Gaussian kernel
        #     kernel_vals = torch.exp(-0.5 * squared_dist)  # Shape: (N,)

        #     # Log the kernel values to check for issues
        #     if torch.isnan(kernel_vals).any() or torch.isinf(kernel_vals).any():
        #         print(f"Kernel values contain NaN or Inf: {kernel_vals}")

        #     # Compute the normalization constant
        #     norm_const = (self.bandwidth ** self.d) * (2 * torch.pi) ** (self.d / 2)

        #     # Log the normalization constant to ensure it's reasonable
        #     if math.isnan(norm_const) or math.isinf(norm_const):
        #         print(f"Normalization constant is problematic: {norm_const}")

        #     # Estimate the density
        #     density = kernel_vals.sum() / (self.n * norm_const)

        #     # Log the density before taking the log
        #     if torch.isnan(density) or torch.isinf(density):
        #         print(f"Density is NaN or Inf: {density}")
            
        #     # Avoid log(0) by adding a small epsilon
        #     epsilon = 1e-10
        #     density_safe = density + epsilon
            
        #     # Log the safe density value
        #     if torch.isnan(density_safe) or torch.isinf(density_safe):
        #         print(f"Density after adding epsilon is problematic: {density_safe}")

        #     # Compute log likelihood
        #     log_likelihood = torch.log(density_safe)
            
        #     # Log the log-likelihood to check for issues
        #     if torch.isnan(log_likelihood).any() or torch.isinf(log_likelihood).any():
        #         print(f"Log-likelihood contains NaN or Inf: {log_likelihood}")

        #     ood_score = -1.0 * (log_likelihood.mean())

        #     # Final check on the OOD score
        #     if torch.isnan(ood_score) or torch.isinf(ood_score):
        #         print(f"OOD score is NaN or Inf: {ood_score}")

        else:
            raise NotImplementedError
        
        if current_img is not None:
            if isinstance(current_img, np.ndarray):
                img = self.images[idx]
                img = img.squeeze(0)
                img = img.detach().cpu().numpy()
                image_data = img * 255  # Generate a random color image
                image_data = image_data.astype(np.uint8)        # Convert to unsigned 8-bit
                image_data = np.transpose(image_data, (1, 2, 0))
                if image_data.shape[2] == 3:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite('nearest_ngb_2.png', image_data)
                self.nn_list.append(image_data)
                # print('current_img ', current_img.shape)
                current_img = current_img * 255  # Generate a random color image
                current_img = current_img.squeeze(0)
                current_img = current_img.astype(np.uint8)        # Convert to unsigned 8-bit
                current_img = np.transpose(current_img, (1, 2, 0))
                if current_img.shape[2] == 3:
                    # Optionally convert from RGB to BGR (since OpenCV uses BGR)
                    current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('current_img_2.png', current_img)
                self.current_image_list.append(current_img)
            elif isinstance(current_img, dict):
                nn_img = {}
                for key in current_img.keys():
                    nn_img[key] = self.images[key][idx].clone()
                    nn_img[key] = nn_img[key].squeeze(0).squeeze(0).detach().cpu().numpy()
                    nn_img[key] = (nn_img[key] * 255).astype(np.uint8)    # Generate a random color image
                    nn_img[key] = np.transpose(nn_img[key], (1, 2, 0))
                    if nn_img[key].shape[2] == 3:
                        nn_img[key] = cv2.cvtColor(nn_img[key], cv2.COLOR_RGB2BGR)
                stacked_nn_img = np.vstack([nn_img[key] for key in nn_img.keys()])
                cv2.imwrite(f'nearest_ngb_{self.image_timestep}.png', stacked_nn_img)
                self.nn_list.append(stacked_nn_img)

                stacked_current_img = np.vstack([current_img[key] for key in current_img.keys()])
                cv2.imwrite(f'current_img_{self.image_timestep}.png', stacked_current_img)
                self.current_image_list.append(stacked_current_img)

        return ood_score

    def compute_ood_score(self, eval_latents, current_img=None):
        if len(eval_latents.shape) == 3:
            eval_latents = eval_latents.reshape(eval_latents.size(0), -1)
        eval_latents = F.normalize(eval_latents, p=2, dim=-1)
        distances = torch.cdist(eval_latents, self.train_latents, p=2)  # (batch_size, num_train_samples)
        min_distances = distances.min(dim=1, keepdim=True)[0]       
        
        reverse_distances = torch.cdist(self.train_latents, eval_latents, p=2)  # L2 (Euclidean) distance
        # Find the minimum distance for each training latent and its corresponding eval latent
        min_reverse_distances, _ = torch.min(reverse_distances, dim=1)  # Get min distance for each training latent
        # Find the training latent index with the overall smallest minimum distance
        idx = torch.argmin(min_reverse_distances)

        nn_img = {}
        for key in current_img.keys():
            nn_img[key] = self.images[key][idx].clone()
            nn_img[key] = nn_img[key].squeeze(0).squeeze(0).detach().cpu().numpy()
            nn_img[key] = (nn_img[key] * 255).astype(np.uint8)    # Generate a random color image
            nn_img[key] = np.transpose(nn_img[key], (1, 2, 0))
            if nn_img[key].shape[2] == 3:
                nn_img[key] = cv2.cvtColor(nn_img[key], cv2.COLOR_RGB2BGR)
        stacked_nn_img = np.vstack([nn_img[key] for key in nn_img.keys()])
        cv2.imwrite('nearest_ngb_2.png', stacked_nn_img)
        self.nn_list.append(stacked_nn_img)

        stacked_current_img = np.vstack([current_img[key] for key in current_img.keys()])
        cv2.imwrite('current_img_2.png', stacked_current_img)
        self.current_image_list.append(stacked_current_img)
        

        return min_distances
     

    def save_nn_list_as_video(self, save_dir, output_filename='nn_list_video-02-09-48.mp4', fps=30):
        if not self.nn_list:
            raise ValueError("The image list is empty. Cannot save a video.")
        
        height, width, channels = self.nn_list[0].shape
        if channels != 3:
            raise ValueError("Images must have 3 channels (BGR format).")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        output_filename = os.path.join(save_dir, output_filename)
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        for frame in self.nn_list:
            video_writer.write(frame)
        
        # Release the VideoWriter
        video_writer.release()
        print(f"Video saved as {output_filename}")


    def save_current_image_list_as_video(self, save_dir, output_filename='curr_image_video-02-09-48.mp4', fps=30):
        if not self.current_image_list:
            raise ValueError("The image list is empty. Cannot save a video.")
        
        height, width, channels = self.current_image_list[0].shape
        if channels != 3:
            raise ValueError("Images must have 3 channels (BGR format).")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        output_filename = os.path.join(save_dir, output_filename)
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        for frame in self.current_image_list:
            video_writer.write(frame)
        
        # Release the VideoWriter
        video_writer.release()
        print(f"Video saved as {output_filename}")

    def silverman_bandwidth(self,):
        # Compute the standard deviation across each dimension
        sigma = torch.std(self.train_latents, dim=0).mean().item()  # Average std across dimensions
        # Apply Silverman's rule
        h = (4 / (self.d + 2))**(1 / (self.d + 4)) * sigma * self.n**(-1 / (self.d + 4))
        return h


    def ood_score_contrastive(self, latent_eval, temperature=0.07):
        # Normalize embeddings to unit norm
        latent_eval = F.normalize(latent_eval, p=2, dim=-1)
        distances = torch.cdist(latent_eval, self.train_latents, p=2)  # (batch_size, num_train_samples)
        weights = F.softmax(-distances / temperature, dim=1)  # (batch_size, num_train_samples)
        ood_scores = (weights * distances).sum(dim=1)  # (batch_size,)
        return ood_scores
    
    # # dynamic thresholding loss
    # def forward(self, latent_eval, threshold_factor=0.7):
    #     # Compute distances to all training points
    #     distances = torch.cdist(latent_eval, self.train_latents)
    #     # Get the minimum distance for each point in latent_eval
    #     min_distances, _ = distances.min(dim=1)
    #     # Compute a dynamic threshold based on the training data's spread
    #     dynamic_threshold = threshold_factor * self.training_distances.median()
    #     # Loss is activated only when the min distance exceeds this dynamic threshold
    #     dynamic_loss = torch.relu(min_distances - dynamic_threshold)
    #     return dynamic_loss.mean()
    
    # scaled knn
    # def forward(self, latent_eval, p=1):
    #     # Compute distances to all training points
    #     distances = torch.cdist(latent_eval, self.train_latents)
    #     # Get the minimum distance for each point in latent_eval
    #     min_distances, _ = distances.min(dim=1)
    #     # Apply a scaling function to amplify the gradient
    #     scaled_loss = min_distances ** p  # Using p > 1 to amplify gradients when distances are large
    #     return scaled_loss.mean()

    # margin loss
    # def forward(self, latent_eval, current_img=None, margin=0.00):
    #     # Compute distances to all training points
    #     distances = torch.cdist(latent_eval, self.train_latents)
    #     # Get the minimum distance for each point in latent_eval
    #     min_distances, idx = distances.min(dim=1)
    #     if current_img is not None and current_img.shape[0] == 1:
    #         img = self.images[idx]
    #         img = img.squeeze(0)
    #         img = img.detach().cpu().numpy()
    #         image_data = img * 255  # Generate a random color image
    #         image_data = image_data.astype(np.uint8)        # Convert to unsigned 8-bit
    #         image_data = np.transpose(image_data, (1, 2, 0))
    #         if image_data.shape[2] == 3:
    #             image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite('nearest_ngb.png', image_data)
    #         self.nn_list.append(image_data)
    #     # Save the image
    #     if current_img is not None and current_img.shape[0] == 1:
    #         print('current_img ', current_img.shape)
    #         current_img = current_img * 255  # Generate a random color image
    #         current_img = current_img.squeeze(0)
    #         current_img = current_img.astype(np.uint8)        # Convert to unsigned 8-bit
    #         current_img = np.transpose(current_img, (1, 2, 0))
    #         if current_img.shape[2] == 3:
    #             # Optionally convert from RGB to BGR (since OpenCV uses BGR)
    #             current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite('current_img.png', current_img)
    #         self.current_image_list.append(current_img)
    #     # Apply a margin loss
    #     margin_loss = torch.relu(min_distances - margin)  # Only penalizes if min_distance > margin
    #     return margin_loss.mean()

    def save_videos(self,):
        # save self.current_image_list and self.nn_list as videos
        if len(self.current_image_list) > 0:
            save_images_to_video(self.current_image_list, 'eval.mp4', fps=30)
            save_images_to_video(self.nn_list, 'nn.mp4', fps=30)

class MahalanobisOODModule(nn.Module):
    def __init__(self, mean, covariance):
        super(MahalanobisOODModule, self).__init__()
        self.mean = mean
        self.cov_inv = torch.inverse(covariance)

    def forward(self, z):
        diff = z - self.mean
        mahalanobis_distance = torch.einsum('bi,ij,bj->b', diff, self.cov_inv, diff)
        return 0.005 * mahalanobis_distance



if __name__ == "__main__":
    train_latents = torch.randn(100, 32, requires_grad=True, device='cuda')  # Example training latents
    knn_module = KNN_torch(train_latents)

    latent_eval = torch.randn(10, 32, requires_grad=True, device='cuda')  # Example latent evaluation
    ood_scores = knn_module(latent_eval)

    print(f"total_ood_score.requires_grad: {ood_scores.requires_grad}")
    print(f"total_ood_score.grad_fn: {ood_scores.grad_fn}")
