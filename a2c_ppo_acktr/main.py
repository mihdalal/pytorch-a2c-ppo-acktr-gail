import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
)
from rlkit.core import logger as rlkit_logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.envs.dmc_wrappers import ActionRepeat, NormalizeActions, TimeLimit

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def experiment(variant):
    env_class = variant["env_class"]
    env_kwargs = variant["env_kwargs"]
    seed = variant["seed"]
    args = get_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_dir = os.path.expanduser(rlkit_logger.get_snapshot_dir())
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    device = torch.device("cuda:0")

    if env_class == "microwave":
        env_class_ = KitchenMicrowaveV0
    elif env_class == "kettle":
        env_class_ = KitchenKettleV0
    elif env_class == "slide_cabinet":
        env_class_ = KitchenSlideCabinetV0
    elif env_class == "hinge_cabinet":
        env_class_ = KitchenHingeCabinetV0
    elif env_class == "top_left_burner":
        env_class_ = KitchenTopLeftBurnerV0
    elif env_class == "light_switch":
        env_class_ = KitchenLightSwitchV0
    elif env_class == "microwave_kettle_light_top_left_burner":
        env_class_ = KitchenMicrowaveKettleLightTopLeftBurnerV0
    elif env_class == "hinge_slide_bottom_left_burner_light":
        env_class_ = KitchenHingeSlideBottomLeftBurnerLightV0
    else:
        raise EnvironmentError("invalid env provided")
    envs = make_vec_envs(
        env_class_,
        env_kwargs,
        seed,
        variant["num_processes"],
        variant["rollout_kwargs"]["gamma"],
        rlkit_logger.get_snapshot_dir(),
        device,
        False,
        use_raw_actions=variant["use_raw_actions"],
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": args.recurrent_policy},
    )
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, **variant["algorithm_kwargs"])
    torch.backends.cudnn.benchmark = True
    rollouts = RolloutStorage(
        variant["num_steps"],
        variant["num_processes"],
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = (
        int(variant["num_env_steps"])
        // variant["num_steps"]
        // variant["num_processes"]
    )
    num_train_calls = 0
    for j in range(num_updates):
        epoch_start_time = time.time()
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if "ppo" == "acktr" else args.lr,
            )

        for step in range(variant["num_steps"]):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
            # for r in reward:
            #     episode_rewards.append(r)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(next_value, **variant["rollout_kwargs"])

        value_loss, action_loss, dist_entropy, num_calls = agent.update(rollouts)
        num_train_calls += num_calls

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (
            j % variant.get("save_interval", int(1e100)) == 0 or j == num_updates - 1
        ) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "ppo")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                os.path.join(save_path, args.env_name + ".pt"),
            )

        if j % variant["log_interval"] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * variant["num_processes"] * variant["num_steps"]
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

        if (
            variant["eval_interval"] is not None
            and len(episode_rewards) > 1
            and j % variant["eval_interval"] == 0
        ):
            evaluate(
                actor_critic,
                envs,
                seed,
                variant["num_processes"],
                eval_log_dir,
                device,
            )
            rlkit_logger.record_tabular(
                "time/epoch (s)", time.time() - epoch_start_time
            )
            rlkit_logger.record_tabular("time/total (s)", time.time() - start)
            rlkit_logger.record_tabular("exploration/num steps total", total_num_steps)
            rlkit_logger.record_tabular("trainer/num train calls", num_train_calls)
            rlkit_logger.record_tabular("Epoch", j // variant["eval_interval"])
            rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)
