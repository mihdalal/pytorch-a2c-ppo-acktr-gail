import numpy as np
import torch
from rad.kitchen_train import compute_path_info
from rlkit.core import logger as rlkit_logger

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, eval_envs, seed, num_processes, eval_log_dir, device):
    # eval_envs = make_vec_envs(
    #     env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True
    # )

    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)
    done = [False] * num_processes
    rewards = 0
    all_infos = []
    while not all(done):
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.append(infos)

        rewards += reward
    mean_ep_reward = rewards.sum().item() / num_processes
    rlkit_logger.record_dict(dict(AverageReturn=mean_ep_reward), prefix="evaluation/")
    statistics = compute_path_info(all_infos)
    rlkit_logger.record_dict(statistics, prefix="evaluation/")
    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            num_processes, mean_ep_reward
        )
    )
