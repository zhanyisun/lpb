import os
from diffusion_policy.common.language_models import extract_text_features
from dyn_model.models.language_encoder import LanguageEncoder
from dyn_model.models.resnet_encoder import ResNetEncoder
import hydra
import torch
import wandb
import logging
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from pathlib import Path
from collections import OrderedDict
import random

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dyn_model_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                if 'tool_hang' in self.cfg.env.train_data_path:
                    project_name = "tool_hang_ph_demo_v141_dyn_model"
                elif 'square' in self.cfg.env.train_data_path:
                    project_name = "square_ph_demo_v141_dyn_model"
                elif 'transport' in self.cfg.env.train_data_path:
                    project_name = "transport_ph_demo_v141_dyn_model"
                elif 'pusht' in self.cfg.env.train_data_path:
                    project_name = "pusht_dyn_model"
                elif 'libero' in self.cfg.env.train_data_path:
                    project_name = "libero_dyn_model"
                self.wandb_run = wandb.init(
                    project=project_name,
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        # Get dataset class from env config
        dataset_class_path = self.cfg.env.dataset_class
        module_path, class_name = dataset_class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        DatasetClass = getattr(module, class_name)
        
        # Common dataset parameters
        dataset_params = {
            'num_hist': self.cfg.num_hist,
            'num_pred': self.cfg.num_pred,
            'frameskip': self.cfg.frameskip,
            'view_names': self.cfg.env.view_names,
            'abs_action': self.cfg.abs_action,
            'use_crop': self.cfg.use_crop,
            'original_img_size': self.cfg.env.original_img_size,
            'cropped_img_size': self.cfg.env.cropped_img_size,
            'action_dim': self.cfg.env.action_dim,
        }
        
        # Add shape_obs for Robomimic and Libero datasets
        if 'RobomimicImageDynamicsModelDataset' in dataset_class_path:
            dataset_params['shape_obs'] = self.cfg.env.shape_obs
        elif 'LiberoImageDynamicsModelDataset' in dataset_class_path:
            dataset_params['shape_obs'] = self.cfg.env.shape_obs
        
        # Create train dataset
        train_dataset = DatasetClass(
            zarr_path=self.cfg.env.train_data_path,
            train=True,
            **dataset_params
        )
        
        # Create validation dataset
        valid_dataset = DatasetClass(
            zarr_path=self.cfg.env.val_data_path,
            train=False,
            **dataset_params
        )
        
        # Set cropped image size from config
        self.cropped_image_size = self.cfg.env.cropped_img_size

        self.original_img_size = train_dataset.original_img_size
        self.normalizer = train_dataset.get_normalizer().to(self.device)
        self.train_img_transform = train_dataset.transform
        self.valid_img_transform = valid_dataset.transform
        state_dict = self.normalizer.state_dict()
        torch.save(state_dict, os.path.join(cfg['saved_folder'], "normalizer.pth"))

        self.datasets = {'train': train_dataset, 'valid': valid_dataset}

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=None,
                prefetch_factor=8
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.language_encoder =None
        self.train_encoder = self.cfg.model.train_encoder
        self.use_pretrained_encoder = self.cfg.use_pretrained_encoder
        self.train_predictor = self.cfg.model.train_predictor
        log.info(f"Train encoder, predictor:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}")
        log.info(f"Use pretrained encoder: {self.use_pretrained_encoder}")
        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor
            else []
        )
        if 'libero' in self.cfg.env.train_data_path:
            self._keys_to_save += ["language_encoder"]
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints", exist_ok=True)
            ckpt = {}
            for key in self._keys_to_save:
                if key not in self.__dict__:
                    continue
                item = self.__dict__[key]
                if isinstance(item, torch.nn.Module):
                    if hasattr(item, "module"):
                        item = self.accelerator.unwrap_model(item)
                    ckpt[key] = item.state_dict()
                    # print('saving ', key)
                elif isinstance(item, torch.optim.Optimizer):
                    ckpt[key] = item.state_dict()
                    # print('saving ', key)
                else:
                    ckpt[key] = item
            ckpt["epoch"] = self.epoch
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename, map_location=self.device)
        self.epoch = ckpt.get("epoch", 0)
        for key in self._keys_to_save:
            if key in ckpt and key in self.__dict__:
                if hasattr(self.__dict__[key], "load_state_dict"):
                    self.__dict__[key].load_state_dict(ckpt[key])
                else:
                    self.__dict__[key] = ckpt[key]
                print('loading ', key)
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            assert self.cfg.env.policy_ckpt_path is not None and \
                self.cfg.encoder_ckpt_path is None and \
                self.train_encoder is False, "Use pretrained encoder with resnet encoder"
            self.encoder = ResNetEncoder(
                policy_ckpt_path=self.cfg.env.policy_ckpt_path,
                view_names=self.cfg.env.view_names,
            )
        encoder_ckpt = None
        if self.cfg.encoder_ckpt_path is not None:
            encoder_ckpt = torch.load(self.cfg.encoder_ckpt_path, map_location=self.device)
            if "encoder" in encoder_ckpt:
                self.encoder.load_state_dict(encoder_ckpt["encoder"])
                log.info(f"Loaded visual encoder from {self.cfg.encoder_ckpt_path}")
            else:
                raise ValueError(f"encoder not found in {self.cfg.encoder_ckpt_path}")
            
        # if not self.train_encoder:
        if self.use_pretrained_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if self.train_encoder:
                # 3. Unfreeze the final layer norm
                for param in self.encoder.base_model.norm.parameters():
                    param.requires_grad = True
                # 4. Unfreeze the (currently Identity) head or your new classification head
                for param in self.encoder.base_model.head.parameters():
                    param.requires_grad = True
                for block_idx in range(11, 12):  # blocks 10..11
                    for param in self.encoder.base_model.blocks[block_idx].parameters():
                        param.requires_grad = True
        language_emb_dim = 0
        if 'libero' in self.cfg.env.train_data_path:
            self.language_encoder = LanguageEncoder(policy_ckpt_path=self.cfg.env.policy_ckpt_path)
            for param in self.language_encoder.parameters():
                param.requires_grad = False
            self.language_encoder = self.accelerator.prepare(self.language_encoder)
            language_emb_dim = 32

        predictor_ckpt = None
        predictor_ckpt_path = self.cfg.predictor_ckpt_path
        if predictor_ckpt_path is not None:
            predictor_ckpt = torch.load(predictor_ckpt_path, map_location=self.device)
        
        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.env.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if predictor_ckpt is not None and "proprio_encoder" in predictor_ckpt:
            self.proprio_encoder.load_state_dict(predictor_ckpt["proprio_encoder"])
            log.info(f"Loaded proprio encoder from {predictor_ckpt_path}")

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.env.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)
        if predictor_ckpt is not None and "action_encoder" in predictor_ckpt:
            self.action_encoder.load_state_dict(predictor_ckpt["action_encoder"])
            log.info(f"Loaded action encoder from {predictor_ckpt_path}")

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        if self.predictor is None:
            self.predictor = hydra.utils.instantiate(
                self.cfg.predictor,
                num_patches=1,
                num_frames=self.cfg.num_hist,
                dim=self.encoder.emb_dim * len(self.cfg.env.view_names)
                + (proprio_emb_dim + action_emb_dim + language_emb_dim),
                visual_dim=self.encoder.emb_dim * len(self.cfg.env.view_names),
                proprio_dim=proprio_emb_dim,
                action_dim=action_emb_dim,
            )
            if predictor_ckpt is not None:
                if "predictor" in predictor_ckpt:
                    self.predictor.load_state_dict(predictor_ckpt["predictor"])
                    log.info(f"Loaded predictor from {self.cfg.predictor_ckpt_path}")
                else:
                    raise ValueError(f"predictor not found in {self.cfg.predictor_ckpt_path}")

        if not self.train_predictor:
            for param in self.predictor.parameters():
                param.requires_grad = False

        self.encoder, self.predictor = self.accelerator.prepare(
            self.encoder, self.predictor
        )

        if not self.train_encoder:
            self.encoder.eval()

        trainable_params = {
            "visual encoder": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            "proprio encoder": sum(p.numel() for p in self.proprio_encoder.parameters() if p.requires_grad),
            "action encoder": sum(p.numel() for p in self.action_encoder.parameters() if p.requires_grad),
        }

        trainable_params["predictor"] = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)

        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            view_names=self.cfg.env.view_names,
            use_layernorm=self.cfg.use_layernorm,
            language_encoder=self.language_encoder,
        )
        # Print results
        for module, count in trainable_params.items():
            print(f"{module}: {count:,} trainable parameters")

    def init_optimizers(self):
        self.encoder_optimizer = None
        if self.model.train_encoder:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if len(encoder_params) > 0:
                self.encoder_optimizer = torch.optim.Adam(
                    encoder_params,
                    lr=self.cfg.training.encoder_lr,
                )
                self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.model.train_predictor:
            predictor_params = [p for p in self.predictor.parameters() if p.requires_grad]
            self.predictor_optimizer = torch.optim.AdamW(
                predictor_params,
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

    def run(self):
        init_epoch = self.epoch + 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            # if i == 1: break
            obs, act, state = data
            for view_name in self.cfg.env.view_names:
                obs['visual'][view_name] = self.normalizer[view_name].normalize(obs['visual'][view_name])
                obs['visual'][view_name] = torch.stack([self.train_img_transform(img) for img in obs['visual'][view_name]])
                obs['visual'][view_name] = obs['visual'][view_name].view(-1, self.cfg.num_hist+self.cfg.num_pred, 3, self.cropped_image_size, self.cropped_image_size)


            obs['proprio'] = self.normalizer['state'].normalize(obs['proprio'])

            if 'language' in obs and obs['language'] is not None:
                language_goal = obs["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                obs["language"] = extract_text_features(
                    self.language_encoder,
                    text_tokens,
                    language_emb_model='clip',
                )

            act = self.normalizer['act'].normalize(act)
            act = rearrange(act, "b (n f) d -> b n (f d)", n=self.cfg.num_hist+self.cfg.num_pred, d=self.datasets['train'].original_action_dim)  # concat actions
            act[:, -1:, :] = 0
            state = self.normalizer['state'].normalize(state)
            self.model.train()
            with self.accelerator.autocast():
                loss, loss_components = self.model(obs, act)
            if self.model.train_encoder:
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()
            if self.model.train_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()
            if self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            # if i == 1: break
            obs, act, state = data
            for view_name in self.cfg.env.view_names:
                obs['visual'][view_name] = self.normalizer[view_name].normalize(obs['visual'][view_name])
                obs['visual'][view_name] = self.valid_img_transform(obs['visual'][view_name].view(-1, 3, self.original_img_size, self.original_img_size))
                obs['visual'][view_name] = obs['visual'][view_name].view(-1, self.cfg.num_hist+self.cfg.num_pred, 3, self.cropped_image_size, self.cropped_image_size)

            obs['proprio'] = self.normalizer['state'].normalize(obs['proprio'])

            if 'language' in obs and obs['language'] is not None:
                language_goal = obs["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                obs["language"] = extract_text_features(
                    self.language_encoder,
                    text_tokens,
                    language_emb_model='clip',
                )

            act = self.normalizer['act'].normalize(act)
            act = rearrange(act, "b (n f) d -> b n (f d)", n=self.cfg.num_hist+self.cfg.num_pred, d=self.datasets['train'].original_action_dim)  # concat actions
            act[:, -1:, :] = 0
            state = self.normalizer['state'].normalize(state)
            self.model.eval()
            with torch.no_grad():
                loss, loss_components = self.model(obs, act)

            loss = self.accelerator.gather_for_metrics(loss.detach().cpu()).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.detach().cpu().mean().item() for key, value in loss_components.items()
            }

            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
