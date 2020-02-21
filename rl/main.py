import copy
import glob
import os
import time
import pdb
import sys
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from a2c_ppo_acktr.arguments import get_args
home_path = os.getcwd()
home_path = '/'.join(home_path.split('/')[:-2])

sys.path.append(home_path+'/vh_multiagent_models')
import utils_rl_agent as utils_rl_agent

def main():
    args = get_args()
    simulator_type = args.simulator_type

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    experiment_name = 'env.{}/algo{}-gamma{}-sim{}'.format(
            args.env_name, args.algo, args.gamma, args.simulator_type)
    log_dir = os.path.expanduser('{}/{}'.format(args.log_dir, experiment_name))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    tensorboard_writer = None


    if args.tensorboard_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        ts_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, experiment_name, ts_str))


    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, simulator_type, args.seed, args.num_processes,
            args.gamma, args.log_dir, device, False, num_frame_stack=args.num_frame_stack)






    ## ------------------------------------------------------------------------------
    ## Preparing the goal
    ## ------------------------------------------------------------------------------
    if 'virtualhome' in args.env_name:
        pass
        #graph = envs.get_graph()
        #glasses_id = [node['id'] for node in graph['nodes'] if 'wineglass' in node['class_name']]
        #table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
        #goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id][:2]
        #task_goal = {}
        #for i in range(2):
        #    task_goal[i] = goals





    ## ------------------------------------------------------------------------------
    ## Model
    ## ------------------------------------------------------------------------------

    actor_critic = Policy(
        envs.observation_space,
        envs.action_space,
        agent_helper=utils_rl_agent.GraphHelper(simulator_type=simulator_type),
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              tuple([obs_sp.shape for obs_sp in envs.observation_space]), envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    if 'virtualhome' in args.env_name:
        obs = envs.reset() # (graph, task_goal) # torch.Size([1, 4, 84, 84])
    else:
        obs = envs.reset()
    
    for it in range(len(obs)):
        # TODO: movidy
        rollouts.obs[it][0].copy_(obs[it])
    rollouts.to(device)

    
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    [ob[step] for ob in rollouts.obs], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])


            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # obs: tensor, [1, 4, 84, 84],  |       [2, 4, 84, 84]
            # reward: tensor, [1, 1]        |       [2, 1]
            # done: array, array([False])   |       array([False, False])
            # infos: [{'ale.lives': 0}]     |       ({'ale.lives': 0}, {'ale.lives': 0})

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if 'virtualhome' in args.env_name:
                    episode_rewards.append(reward) # info['episode']['r'])
        

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                [ob[-1] for ob in rollouts.obs], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    [ob for ob[step] in rollouts.obs], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, experiment_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "{}_{}.pt".format(args.env_name, j)))

        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("mean_reward", np.mean(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("median_reward", np.median(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("min_reward", np.min(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("max_reward", np.max(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("dist_entropy", dist_entropy, total_num_steps)
                tensorboard_writer.add_scalar("value_loss", value_loss, total_num_steps)
                tensorboard_writer.add_scalar("action_loss", action_loss, total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
