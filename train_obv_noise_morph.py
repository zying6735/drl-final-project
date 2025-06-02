import os
import argparse
from copy import deepcopy
from collections import deque
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64) # Ensure this is intended for all tensors
import torch.nn.functional as F
from torch.distributions.normal import Normal
import wandb
import imageio # Still imported, but video logging will be off by default

from halfcheetah_morph_env import HalfCheetahMorphEnv

# --- Helper to add observation noise ---
def add_observation_noise(observation, noise_std):
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=observation.shape).astype(observation.dtype)
        return observation + noise
    return observation

# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, o, a, r, o_1): # o and o_1 will be augmented and potentially noisy observations
         self.buffer.append((o, a, r, o_1))

    def sample(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(O), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float64, device=self.device).unsqueeze(1),\
               torch.tensor(np.array(O_1), dtype=torch.float64, device=self.device)

    def __len__(self):
        return len(self.buffer)

# --- Critic Network (Q-function) ---
class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size + action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.action_size = action_size

    def forward(self, x, a):
        if x.ndim > 1 and a.ndim == 1 and a.shape[0] == self.action_size :
             a = a.unsqueeze(0).repeat(x.shape[0], 1)
        elif x.ndim == 1 and a.ndim > 1 and a.shape[0] == 1:
            a = a.squeeze(0)
        elif x.ndim > 1 and a.ndim > 1 and x.shape[0] != a.shape[0]:
            if a.shape[0] == 1 and x.shape[0] > 1:
                a = a.repeat(x.shape[0], 1)
            else:
                raise ValueError(f"Shape mismatch in Q_FC: x.shape={x.shape}, a.shape={a.shape}")
        combined_input = torch.cat((x, a), dim=-1)
        y1 = F.relu(self.fc1(combined_input))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)
        return y

# --- Actor Network (Policy) ---
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu_layer = torch.nn.Linear(256, action_size)
        self.log_sigma_layer = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu_layer(y2)
        log_prob = None
        if deterministic:
            action = torch.tanh(mu)
        else:
            log_sigma = self.log_sigma_layer(y2)
            log_sigma = torch.clamp(log_sigma, min=-20.0, max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            if with_logprob:
                log_prob_u = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
                action_tanh = torch.tanh(x_t)
                log_prob = log_prob_u - torch.log(1.0 - action_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            action = torch.tanh(x_t)
        return action, log_prob

# --- Soft Actor-Critic (SAC) Agent ---
class SAC:
    def __init__(self, arglist):
        self.arglist = arglist

        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)
        if torch.cuda.is_available() and self.arglist.use_gpu: # Added use_gpu check
            torch.cuda.manual_seed_all(self.arglist.seed)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = os.path.join(script_dir, "half_cheetah.xml")
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_path}.")

        try:
            with HalfCheetahMorphEnv(xml_path=self.xml_path, use_custom_morphology=False, render_mode=False) as temp_env:
                self.base_obs_size = temp_env.observation_space.shape[0]
                self.action_size = temp_env.action_space.shape[0]
        except Exception as e:
            print(f"Error creating temporary environment to determine base_obs_size: {e}")
            raise

        # Determine if agent uses augmented observations (manual concatenation)
        self.agent_uses_augmented_obs = self.arglist.manual_obs_augmentation
        if self.agent_uses_augmented_obs:
            self.num_morph_params = 4
            self.obs_size = self.base_obs_size + self.num_morph_params
            self.morph_param_keys = ['b_thigh_len', 'b_shin_len', 'f_thigh_len', 'f_shin_len']
            print(f"Agent uses MANUALLY AUGMENTED observations.")
            print(f"Base obs_size: {self.base_obs_size}, Num morph_params: {self.num_morph_params}, Total (augmented) obs_size: {self.obs_size}")
        else:
            # This case assumes the environment itself provides the full observation
            # including any morphology parameters if they are part of the state.
            # For the original script (no manual concat), self.obs_size is just base_obs_size.
            self.obs_size = self.base_obs_size
            print(f"Agent uses observations directly from ENV. Obs_size: {self.obs_size}")
        print(f"Action_size: {self.action_size}")


        self.max_episode_steps = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.arglist.use_gpu else "cpu")
        print(f"Using device: {self.device}")

        self.actor = Pi_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_1 = Q_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_target_1 = deepcopy(self.critic_1).to(self.device)
        self.critic_2 = Q_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_target_2 = deepcopy(self.critic_2).to(self.device)

        for param in self.critic_target_1.parameters(): param.requires_grad = False
        for param in self.critic_target_2.parameters(): param.requires_grad = False

        self.target_entropy = -float(self.action_size)

        log_base_dir = self.arglist.log_dir # This will use the new default from parse_args
        self.exp_dir = os.path.join(log_base_dir, f"seed_{self.arglist.seed}_{self.arglist.wandb_run_name_suffix}")
        self.model_dir = os.path.join(self.exp_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.total_steps = 0

        if self.arglist.resume:
            self._load_checkpoint()
        else:
            self.start_episode = 0
            self.log_alpha = torch.tensor(np.log(self.arglist.initial_alpha), dtype=torch.float64, device=self.device, requires_grad=True)
            self._initialize_optimizers()
            self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
            print("Starting training from scratch.")

        if self.arglist.mode == "train":
            self.train()

    def _initialize_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.arglist.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.arglist.lr)

    def _augment_observation_if_needed(self, raw_obs, leg_lengths_dict):
        if self.agent_uses_augmented_obs:
            leg_values = np.array([leg_lengths_dict[key] for key in self.morph_param_keys], dtype=np.float64)
            if raw_obs.dtype != np.float64:
                raw_obs = raw_obs.astype(np.float64)
            return np.concatenate((raw_obs, leg_values))
        return raw_obs.astype(np.float64) # Ensure correct dtype even if not augmenting

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(self.model_dir, "backup.ckpt")
        if not os.path.exists(checkpoint_path):
            print(f"Resume: Checkpoint {checkpoint_path} not found. Starting fresh.")
            self.start_episode = 0
            self.total_steps = 0
            self.log_alpha = torch.tensor(np.log(self.arglist.initial_alpha), dtype=torch.float64, device=self.device, requires_grad=True)
            self._initialize_optimizers()
            self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
            return
        try:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_episode = checkpoint['episode'] + 1
            self.total_steps = checkpoint.get('total_steps', 0)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic_1.load_state_dict(checkpoint['critic_1'])
            self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
            self.critic_2.load_state_dict(checkpoint['critic_2'])
            self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
            log_alpha_value = checkpoint['log_alpha']
            if isinstance(log_alpha_value, torch.Tensor):
                self.log_alpha = log_alpha_value.clone().detach().requires_grad_(True).to(self.device)
            else:
                self.log_alpha = torch.tensor(log_alpha_value, dtype=torch.float64, device=self.device, requires_grad=True)
            self._initialize_optimizers()
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1'])
            self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
            self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
            if 'replay_buffer_content' in checkpoint and checkpoint['replay_buffer_content'] is not None:
                for item in checkpoint['replay_buffer_content']:
                    self.replay_buffer.push(*item)
            print(f"Resumed from episode {self.start_episode}, total_steps {self.total_steps}.")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            if len(self.replay_buffer) > 0:
                buffer_obs_dim = self.replay_buffer.buffer[0][0].shape[0]
                if buffer_obs_dim != self.obs_size:
                    print(f"WARNING: Replay buffer obs dim ({buffer_obs_dim}) != agent obs_size ({self.obs_size}). Check config.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            self.start_episode = 0
            self.total_steps = 0
            self.log_alpha = torch.tensor(np.log(self.arglist.initial_alpha), dtype=torch.float64, device=self.device, requires_grad=True)
            self._initialize_optimizers()
            self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

    def save_model_checkpoint(self, episode_num):
        checkpoint_content = {'actor' : self.actor.state_dict()}
        save_path = os.path.join(self.model_dir, f"model_ep{episode_num}_steps{self.total_steps}.ckpt")
        torch.save(checkpoint_content, save_path)
        print(f"Saved model checkpoint to {save_path}")

    def save_backup_checkpoint(self, episode_num):
        checkpoint_state = {
            'episode' : episode_num, 'total_steps': self.total_steps,
            'actor' : self.actor.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1' : self.critic_1.state_dict(), 'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
            'critic_2' : self.critic_2.state_dict(), 'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
            'critic_target_1' : self.critic_target_1.state_dict(), 'critic_target_2' : self.critic_target_2.state_dict(),
            'log_alpha' : self.log_alpha.detach().clone(), 'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'replay_buffer_content' : list(self.replay_buffer.buffer) if self.arglist.save_replay_buffer else None
        }
        save_path = os.path.join(self.model_dir, "backup.ckpt")
        torch.save(checkpoint_state, save_path)
        print(f"Saved backup checkpoint to {save_path} at episode {episode_num}, total_steps {self.total_steps}")

    def soft_update(self, target_network, source_network, tau):
        with torch.no_grad():
            for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    # Video logging function remains, but will only be called if arglist.log_video is True
    def log_video_to_wandb(self, episode_num, leg_lengths_dict, floor_friction_list):
        if not self.arglist.log_video and not self.arglist.log_video_local:
            return

        print(f"Generating video for episode {episode_num} (total steps {self.total_steps})")
        local_video_dir = os.path.join(self.exp_dir, "local_videos_exac_train_noise") # Updated suffix
        local_video_path = None
        if self.arglist.log_video_local:
            os.makedirs(local_video_dir, exist_ok=True)
            local_video_path = os.path.join(local_video_dir, f"episode_{episode_num}_steps_{self.total_steps}.gif")

        video_env = None
        collected_frames = []
        try:
            video_env = HalfCheetahMorphEnv(
                xml_path=self.xml_path, leg_lengths=leg_lengths_dict, floor_friction=floor_friction_list,
                use_custom_morphology=True, render_mode=True
            )
            raw_obs_video = video_env.reset()
            obs_for_video_actor = self._augment_observation_if_needed(raw_obs_video, leg_lengths_dict)
            # Note: Video generation does not add training noise to observations by default
            # If desired, noise could be added here too, but typically eval/viz is done on clean obs.

            for _step in range(self.max_episode_steps):
                with torch.no_grad():
                    o_tensor = torch.tensor(obs_for_video_actor, dtype=torch.float64, device=self.device).unsqueeze(0)
                    a_tensor, _ = self.actor(o_tensor, deterministic=True)
                action = a_tensor.cpu().numpy().squeeze()
                raw_next_obs_video, _, env_done, _ = video_env.step(action.astype(np.float32))
                obs_for_video_actor = self._augment_observation_if_needed(raw_next_obs_video, leg_lengths_dict)
                frame = video_env.render(mode='rgb_array')
                if frame is not None: collected_frames.append(frame.copy())
                if env_done: break
            
            if collected_frames:
                video_np = np.array(collected_frames)
                if video_np.ndim == 4 and video_np.shape[0] > 0 and video_np.shape[3] == 3:
                    if video_np.dtype != np.uint8:
                        if np.issubdtype(video_np.dtype, np.floating) and video_np.max() <= 1.0 and video_np.min() >= 0.0:
                            video_np = (video_np * 255).astype(np.uint8)
                        else: video_np = np.clip(video_np, 0, 255).astype(np.uint8)
                    if self.arglist.log_video_local and local_video_path:
                        imageio.mimsave(local_video_path, video_np, fps=self.arglist.video_fps)
                        print(f"  Saved video locally: {local_video_path}")
                    if self.arglist.log_video:
                        wandb.log({"agent_rollout_video": wandb.Video(video_np, fps=self.arglist.video_fps, format="gif")}, step=self.total_steps)
                        print(f"  Logged video to wandb for episode {episode_num}.")
                else: print(f"  Video array has problematic shape: {video_np.shape}. Skipping save/log.")
            else: print(f"No frames for video at ep {episode_num}.")
        except Exception as e_outer: print(f"Error during video generation: {e_outer}")
        finally:
            if video_env: video_env.close()

    def train(self):
        run_name = f"seed_{self.arglist.seed}_{self.arglist.wandb_run_name_suffix}"
        wandb.init(
            project=self.arglist.wandb_project_name, name=run_name,
            config=vars(self.arglist), resume="allow",
            id=run_name if self.arglist.resume else None
        )
        # Log if manual augmentation and training noise std are used
        wandb.config.update({
            "manual_obs_augmentation": self.agent_uses_augmented_obs,
            "train_obs_noise_std": self.arglist.train_obs_noise_std
            }, allow_val_change=True)

        print(f"Starting training for {self.arglist.episodes} episodes with obs noise std: {self.arglist.train_obs_noise_std}.")
        print(f"Logging to wandb project: {self.arglist.wandb_project_name}, run: {run_name}")
        if self.start_episode > 0:
             print(f"Resuming from episode {self.start_episode + 1}, total_steps {self.total_steps}")

        current_leg_lengths = {key: 0.1 for key in (self.morph_param_keys if self.agent_uses_augmented_obs else [])}
        current_floor_friction = [0.4, 0.1, 0.1]
        training_env = None

        for episode_idx in range(self.start_episode, self.arglist.episodes):
            current_episode_num = episode_idx + 1
            if episode_idx == self.start_episode or \
               (episode_idx > self.start_episode and episode_idx % self.arglist.morph_change_every == 0):
                current_leg_lengths = {
                    'b_thigh_len': random.uniform(0.0725, 0.2175), 'b_shin_len': random.uniform(0.075, 0.225),
                    'f_thigh_len': random.uniform(0.0665, 0.1995), 'f_shin_len': random.uniform(0.053, 0.159)
                }
                friction_val = random.uniform(0.2, 0.6)
                current_floor_friction = [friction_val, 0.1, 0.1]
                print(f"\nEp {current_episode_num}: New morph. Legs: {current_leg_lengths}, Friction: {current_floor_friction}")
                wandb.log({
                    "morphology/b_thigh_len": current_leg_lengths['b_thigh_len'],
                    "morphology/b_shin_len": current_leg_lengths['b_shin_len'],
                    "morphology/f_thigh_len": current_leg_lengths['f_thigh_len'],
                    "morphology/f_shin_len": current_leg_lengths['f_shin_len'],
                    "morphology/floor_friction_main": current_floor_friction[0],
                }, step=self.total_steps)

            if training_env is not None: training_env.close()
            try:
                training_env = HalfCheetahMorphEnv(
                    xml_path=self.xml_path, leg_lengths=current_leg_lengths, floor_friction=current_floor_friction,
                    use_custom_morphology=True, render_mode=False
                )
            except Exception as e:
                print(f"FATAL: Could not create training env at ep {current_episode_num}: {e}")
                wandb.finish(exit_code=1); return

            raw_obs_from_env = training_env.reset()
            # Augment if needed, then add noise
            obs_for_agent = self._augment_observation_if_needed(raw_obs_from_env, current_leg_lengths)
            obs_for_agent = add_observation_noise(obs_for_agent, self.arglist.train_obs_noise_std)

            episode_reward = 0; episode_steps = 0
            for t in range(self.max_episode_steps):
                if self.total_steps < self.arglist.start_steps:
                    action = training_env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs_for_agent, dtype=torch.float64, device=self.device).unsqueeze(0)
                        action_tensor, _ = self.actor(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy().squeeze()

                raw_next_obs_from_env, reward, env_terminated, info = training_env.step(action.astype(np.float32))
                # Augment if needed, then add noise for next state
                next_obs_for_agent = self._augment_observation_if_needed(raw_next_obs_from_env, current_leg_lengths)
                next_obs_for_agent = add_observation_noise(next_obs_for_agent, self.arglist.train_obs_noise_std)

                # Store the (potentially noisy) obs_for_agent and next_obs_for_agent in replay buffer
                self.replay_buffer.push(obs_for_agent, action, reward, next_obs_for_agent)
                obs_for_agent = next_obs_for_agent
                episode_reward += reward; episode_steps += 1; self.total_steps += 1

                if len(self.replay_buffer) >= self.arglist.replay_fill and \
                   self.total_steps % self.arglist.update_every == 0:
                    for _ in range(self.arglist.updates_per_step):
                        obs_batch, action_batch, reward_batch, next_obs_batch = self.replay_buffer.sample(self.arglist.batch_size)
                        with torch.no_grad():
                            next_action_batch, next_log_prob_batch = self.actor(next_obs_batch, with_logprob=True)
                            q1_next_target = self.critic_target_1(next_obs_batch, next_action_batch)
                            q2_next_target = self.critic_target_2(next_obs_batch, next_action_batch)
                            min_q_next_target = torch.min(q1_next_target, q2_next_target)
                            alpha = torch.exp(self.log_alpha).detach()
                            target_q_value = reward_batch + self.arglist.gamma * (min_q_next_target - alpha * next_log_prob_batch)
                        q1_current = self.critic_1(obs_batch, action_batch); q2_current = self.critic_2(obs_batch, action_batch)
                        critic_loss_1 = F.mse_loss(q1_current, target_q_value); critic_loss_2 = F.mse_loss(q2_current, target_q_value)
                        critic_loss = critic_loss_1 + critic_loss_2
                        self.critic_optimizer_1.zero_grad(); critic_loss_1.backward(); self.critic_optimizer_1.step()
                        self.critic_optimizer_2.zero_grad(); critic_loss_2.backward(); self.critic_optimizer_2.step()
                        for p in self.critic_1.parameters(): p.requires_grad = False
                        for p in self.critic_2.parameters(): p.requires_grad = False
                        pi_action_batch, pi_log_prob_batch = self.actor(obs_batch, with_logprob=True)
                        q1_pi = self.critic_1(obs_batch, pi_action_batch); q2_pi = self.critic_2(obs_batch, pi_action_batch)
                        min_q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = (alpha * pi_log_prob_batch - min_q_pi).mean()
                        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
                        for p in self.critic_1.parameters(): p.requires_grad = True
                        for p in self.critic_2.parameters(): p.requires_grad = True
                        alpha_loss = -(self.log_alpha * (pi_log_prob_batch.detach() + self.target_entropy)).mean()
                        self.log_alpha_optimizer.zero_grad(); alpha_loss.backward(); self.log_alpha_optimizer.step()
                        alpha = torch.exp(self.log_alpha).detach()
                        self.soft_update(self.critic_target_1, self.critic_1, self.arglist.tau)
                        self.soft_update(self.critic_target_2, self.critic_2, self.arglist.tau)
                        if self.total_steps % self.arglist.log_interval == 0:
                            wandb.log({
                                "training/critic_loss": critic_loss.item(), "training/actor_loss": actor_loss.item(),
                                "training/alpha_loss": alpha_loss.item(), "training/alpha": alpha.item(),
                                "training/q1_current_mean": q1_current.mean().item(), "training/q2_current_mean": q2_current.mean().item(),
                                "training/target_q_mean": target_q_value.mean().item(), "training/log_prob_mean": pi_log_prob_batch.mean().item(),
                                "training/replay_buffer_size": len(self.replay_buffer),
                            }, step=self.total_steps)
                if env_terminated: break
            print(f"Ep {current_episode_num}/{self.arglist.episodes} | Reward: {episode_reward:.2f} | Ep Steps: {episode_steps} | Total Steps: {self.total_steps}")
            wandb.log({"episode/reward": episode_reward, "episode/steps_count": episode_steps, "episode_num": current_episode_num}, step=self.total_steps)
            if current_episode_num % self.arglist.save_model_every == 0 or current_episode_num == self.arglist.episodes:
                self.save_model_checkpoint(current_episode_num)
            if current_episode_num % self.arglist.backup_every == 0 or current_episode_num == self.arglist.episodes:
                if not (self.arglist.resume and episode_idx == self.start_episode and self.start_episode !=0):
                     self.save_backup_checkpoint(current_episode_num)
            # Video logging call remains, but controlled by arglist.log_video which defaults to False now
            if current_episode_num % self.arglist.video_log_every == 0:
                self.log_video_to_wandb(current_episode_num, current_leg_lengths, current_floor_friction)
        
        if training_env is not None: training_env.close()
        # Final video log attempt if enabled
        if self.arglist.log_video or self.arglist.log_video_local:
            self.log_video_to_wandb(self.arglist.episodes, current_leg_lengths, current_floor_friction)
        print("Training finished.")
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser("SAC Training with Observation Noise, Custom HalfCheetah, Random Morphology, and WandB Logging")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Execution mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU if available")
    # --- MODIFIED: log_dir for new experiment (training with obs noise) ---
    parser.add_argument("--log_dir", type=str, default="./logs_sac_morph_train_obsnoise", help="Base directory for logs and models")

    # --- NEW: Argument for manual observation augmentation strategy ---
    parser.add_argument("--manual_obs_augmentation", action="store_true", default=True, # Defaulting to True as per your script
                        help="Manually concatenate leg lengths to observation. If False, assumes env provides full state.")

    # --- NEW: Argument for training observation noise ---
    parser.add_argument("--train_obs_noise_std", type=float, default=0.01,
                        help="Standard deviation of Gaussian noise added to observations during training (default: 0.01). Set to 0 for no noise.")

    parser.add_argument("--episodes", type=int, default=5000, help="Total number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--initial_alpha", type=float, default=0.2, help="Initial alpha")
    parser.add_argument("--start_steps", type=int, default=10000, help="Random action steps")
    parser.add_argument("--replay_size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--replay_fill", type=int, default=10000, help="Min transitions before training")
    parser.add_argument("--update_every", type=int, default=1, help="Env steps per training update")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per training step")
    parser.add_argument("--morph_change_every", type=int, default=1, help="Episodes to change morphology")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training")
    parser.add_argument("--save_model_every", type=int, default=250, help="Episode frequency to save model")
    parser.add_argument("--backup_every", type=int, default=100, help="Episode frequency to save backup")
    parser.add_argument("--save_replay_buffer", action="store_true", default=False, help="Save replay buffer in backups")
    parser.add_argument("--log_interval", type=int, default=1000, help="Total_steps frequency to log metrics")

    # --- MODIFIED: W&B Naming for new experiment ---
    parser.add_argument("--wandb_project_name", type=str, default="SAC_HC_TrainObsNoise_exac", help="WandB project name")
    parser.add_argument("--wandb_run_name_suffix", type=str, default="train_obsnoise_exac", help="Suffix for WandB run name")

    # --- MODIFIED: Video logging off by default ---
    parser.add_argument("--log_video", action=argparse.BooleanOptionalAction, default=False, help="Log videos to WandB. Use --log-video to enable.")
    parser.add_argument("--log_video_local", action=argparse.BooleanOptionalAction, default=False, help="Log videos locally. Use --log-video-local to enable.")
    parser.add_argument("--video_log_every", type=int, default=250, help="Frequency (episodes) to log videos IF enabled (e.g., every 250 episodes)")
    parser.add_argument("--video_fps", type=int, default=30, help="FPS for logged videos")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    # The SAC class init now considers args.manual_obs_augmentation
    sac_agent = SAC(args)