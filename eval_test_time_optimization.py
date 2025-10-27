import sys
import os
import pathlib

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

@hydra.main(config_path="dyn_model/conf/planner", config_name="eval_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir

    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # Load policy_checkpoint
    with open(cfg.policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    
    # Update configuration based on payload
    cfg_task_env_runner = payload['cfg']
    cfg_task_env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.task.env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.policy.n_action_steps = cfg.n_action_steps

    cfg_task_env_runner.task.env_runner.n_test = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_test_vis = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_train = 0
    cfg_task_env_runner.task.env_runner.n_train_vis = 0
    cfg_task_env_runner.task.env_runner.test_start_seed = cfg.test_start_seed

    if 'libero' in cfg.policy_checkpoint:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.dataset_path
        
    # Initialize workspace
    cls = hydra.utils.get_class(cfg_task_env_runner._target_)
    workspace = cls(cfg_task_env_runner, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace
    policy = workspace.model
    if cfg_task_env_runner.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()
    
    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=payload['cfg'].task.dataset,
        dynamics_model_ckpt=cfg.dynamics_model_checkpoint,
        action_step=cfg_task_env_runner.n_action_steps,
        output_dir=cfg.output_dir,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.get('demo_dataset_path', None)
    )

    # Run evaluation - use env_runner_target from the planner config
    cfg_task_env_runner.task.env_runner._target_ = cfg.env_runner_target

    # Check if it's a libero task by examining the dataset target
    dataset_target = payload['cfg'].task.dataset._target_
    if 'libero' in dataset_target:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
            task_dir=cfg_task_env_runner.task.env_runner.dataset_path
        )
    else:
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir
        )

    runner_log = env_runner.run(policy)
    
    # Save evaluation results separately
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value
    
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
