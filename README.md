# Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution

[[Project page]](https://project-latentpolicybarrier.github.io/)
[[Paper]](https://arxiv.org/abs/2508.05941)
[[Data and Models]](https://drive.google.com/drive/folders/1cSuUhdwv6bQpaSIHz15mZz-V_ZuGhzvc?usp=sharing)

[Zhanyi Sun](https://zhanyisun.github.io/),
[Shuran Song](https://shurans.github.io/)

## Installation
Install conda environment with 
```console
$ mamba env create -f conda_environment.yaml
```
or 
```console
$ conda env create -f conda_environment.yaml
```

## Reproducing Simulation Results 
### Base Diffusion Policy Training

Under the repository root, create a `data/` subdirectory to store all task datasets.  
Download the corresponding demonstration data and place it inside the `data/` folder.

- **PushT Task:** Use the demonstration dataset provided by the [Diffusion Policy codebase](https://github.com/real-stanford/diffusion_policy).
- **Robomimic / Libero Tasks:** Use the demonstration datasets provided by their respective official repositories.

For example, to train a base Diffusion Policy on the **Transport** task: first download the dataset from [this link](https://drive.google.com/drive/folders/1QvYUMCewm9XKcQ1Vy2yj4_cGtCaIPjQ8?usp=drive_link). Then place it under `data/transport/data/expert_demonstration/`. Run the training command as described below
```console
(lpb)[lpb]$ python train.py --config-dir=. --config-name=image_transport_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
This will create an output directory with format `data/outputs/YYYY.MM.DD/HH.MM.SS_${name}_${task_name}`.

To train a multi-task base diffusion policy for **Libero10** tasks, download data from [link](https://drive.google.com/drive/folders/11AweLP_N5OL3Df9CDktYrwMfx_fikYSu?usp=sharing), extract it and place it under `data/libero10/data/expert_demonstration/` subdirectory, then run
```console
(lpb)[lpb]$ python train.py --config-dir=. --config-name=image_transport_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### Dynamica Model Training

The diffusion base policy training allows saving policy checkpoints at desired intervals. You can use these saved checkpoints to perform rollouts and generate additional rollout datasets. After aggregating the rollout data with expert demonstrations, you can train a dynamics model on the combined dataset.

We provide example configurations for dynamics model training on the **Transport** and **Libero10** tasks.

- **Transport Task:**
  - Download our pre-collected rollout dataset from [this link](https://drive.google.com/drive/folders/1LdI1eUe_95O7XT9uYLAnx9KC4UNSSF5x?usp=sharing) and place it under `data/transport/data/rollout/` subdirectory. Set the paths to both the combined dataset and the policy checkpoint in `dyn_model/conf/train.yaml`/ 
  - Since the dynamics model reuses the encoder from the base policy checkpoint, ensure that the path to the pretrained base policy checkpoint is also specified in the same config file. We provide a pretrained base policy checkpoint for the Transport task at [this link](https://drive.google.com/drive/folders/1zsnjwRAZOQQYE9eIg64Fxl7AK11aw9Tu?usp=sharing), but you are also welcome to train your own.

After setting the paths to the rollout dataset and pretrained base policy checkpoint,  
update the `env_name` in `dyn_model/conf/train.yaml` to `"transport"`,  
then start the dynamics model training:

```console
(lpb)[lpb]$ python dyn_model/train.py --config-name train.yaml
```
This will create an output directory with format `data/outputs/YYYY.MM.DD/HH.MM.SS_${env_name}`.

- **Libero10 Task:**
  - Download our pre-collected rollout dataset from [this link](https://drive.google.com/drive/folders/1pIjNS9bbHTDfiu3NFG9Qw9G5ostVmxZi?usp=sharing) and place it under `data/libero10/data/rollout/` subdirectory. Set the paths to both the combined dataset and the policy checkpoint in `dyn_model/conf/train.yaml`/ 
  - Download pretrained base policy checkpoint for the Libero10 task at [this link](https://drive.google.com/drive/folders/1GLrkM7ulRcZYaN7_a20M-JC2qdtfvtk9?usp=sharing). You are also welcome to train your own.

After setting the paths to the rollout dataset and pretrained base policy checkpoint,  
update the `env_name` in `dyn_model/conf/train.yaml` to `"libero"`,  
then start the dynamics model training with the same training command:
```console
(lpb)[lpb]$ python dyn_model/train.py --config-name train.yaml
```

### Inference with Action Optimization
To run test-time action optimization, you will need a **pretrained diffusion policy checkpoint**, a **pretrained dynamics model checkpoint**, and the **reference expert demonstration dataset** (same one as used by base policy training). We provide pretrained policy and dynamics model checkpoints for directly evaluating test-time optimization. For the **Transport** task, you can download the pretrained diffusion policy checkpoint from [this link](https://drive.google.com/drive/folders/1zsnjwRAZOQQYE9eIg64Fxl7AK11aw9Tu?usp=drive_link) and the pretrained dynamics model from [this link](https://drive.google.com/drive/folders/11mszNq729jNZYQuofGFMQwBlqNmAKghI?usp=drive_link). The expert demonstration dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1QvYUMCewm9XKcQ1Vy2yj4_cGtCaIPjQ8?usp=drive_link). After downloading these files, specify the paths to the corresponding models and dataset in `dyn_model/conf/planner/eval_transport.yaml`, then run test-time action optimization with the command below.


```console
(lpb)[lpb]$ python eval_test_time_optimization.py --config-name=eval_transport
```

For the **Libero10** tasks, you can download the pretrained diffusion policy checkpoint from [this link](https://drive.google.com/drive/folders/1GLrkM7ulRcZYaN7_a20M-JC2qdtfvtk9?usp=drive_link) and the pretrained dynamics model from [this link](https://drive.google.com/drive/folders/1NPyj9grvmY1vJ9PMUOePleD4LQ5geJU_?usp=sharing). The expert demonstration dataset can be downloaded from [this link](https://drive.google.com/drive/folders/11AweLP_N5OL3Df9CDktYrwMfx_fikYSu?usp=drive_link). After downloading these files, specify the paths to the corresponding models and dataset in `dyn_model/conf/planner/eval_libero.yaml`. For Libero10, evaluation is performed on **one task at a time**, and the task ID is determined by the `dataset_path` argument in the configuration file. For example, setting  
`data/libero10/data/expert_demonstration/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`  
will run the task with ID  
`STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`. After preparing the `dyn_model/conf/planner/eval_libero.yaml` file, you can run test-time action optimization with the command below

```console
(lpb)[lpb]$ python eval_test_time_optimization.py --config-name=eval_libero
```

## Code

 - [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/): The base diffusion policy was built on top of the Diffusion Policy codebase.
 - [DINO-WM](https://github.com/gaoyuezhou/dino_wm): Part of the dynamics model training code was adapted from the DINO-WM codebase.

 If you find this work useful, consider citing:

```bibtex
@article{sun2025latent,
  title={Latent policy barrier: Learning robust visuomotor policies by staying in-distribution},
  author={Sun, Zhanyi and Song, Shuran},
  journal={arXiv preprint arXiv:2508.05941},
  year={2025}
}
```