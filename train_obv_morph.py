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
import imageio

from halfcheetah_morph_env import HalfCheetahMorphEnv

# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, o, a, r, o_1): # o and o_1 will be augmented observations
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
    def __init__(self, obs_size, action_size): # obs_size will be base_obs_size + 4
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
    def __init__(self, obs_size, action_size): # obs_size will be base_obs_size + 4
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.arglist.seed)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = os.path.join(script_dir, "half_cheetah.xml")
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_path}.")

        # --- MODIFIED: Determine base_obs_size and total obs_size ---
        try:
            # Instantiate with default/non-custom morphology to get base observation size
            with HalfCheetahMorphEnv(xml_path=self.xml_path, use_custom_morphology=False, render_mode=False) as temp_env:
                self.base_obs_size = temp_env.observation_space.shape[0]
                self.action_size = temp_env.action_space.shape[0]
        except Exception as e:
            print(f"Error creating temporary environment to determine base_obs_size: {e}")
            raise

        self.num_morph_params = 4 # b_thigh, b_shin, f_thigh, f_shin
        self.obs_size = self.base_obs_size + self.num_morph_params
        # Define the order of leg lengths for consistent augmentation
        self.morph_param_keys = ['b_thigh_len', 'b_shin_len', 'f_thigh_len', 'f_shin_len']
        print(f"Base obs_size: {self.base_obs_size}, Num morph_params: {self.num_morph_params}, Total (augmented) obs_size: {self.obs_size}")
        print(f"Action_size: {self.action_size}")
        # --- END MODIFICATION ---

        self.max_episode_steps = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.arglist.use_gpu else "cpu")
        print(f"Using device: {self.device}")

        # Networks are initialized with the augmented obs_size
        self.actor = Pi_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_1 = Q_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_target_1 = deepcopy(self.critic_1).to(self.device)
        self.critic_2 = Q_FC(self.obs_size, self.action_size).to(self.device)
        self.critic_target_2 = deepcopy(self.critic_2).to(self.device)

        for param in self.critic_target_1.parameters(): param.requires_grad = False
        for param in self.critic_target_2.parameters(): param.requires_grad = False

        self.target_entropy = -float(self.action_size)

        log_base_dir = self.arglist.log_dir
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

    # --- NEW HELPER METHOD ---
    def _augment_observation(self, raw_obs, leg_lengths_dict):
        """
        Concatenates leg length parameters to the raw observation from the environment.
        """
        leg_values = np.array([leg_lengths_dict[key] for key in self.morph_param_keys], dtype=np.float64)
        # Ensure raw_obs is also float64 if it's not already, for consistent concatenation
        if raw_obs.dtype != np.float64:
            raw_obs = raw_obs.astype(np.float64)
        augmented_obs = np.concatenate((raw_obs, leg_values))
        return augmented_obs
    # --- END NEW HELPER METHOD ---

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
                    self.replay_buffer.push(*item) # item already contains augmented obs if saved from this setup

            print(f"Resumed from episode {self.start_episode}, total_steps {self.total_steps}.")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            if len(self.replay_buffer) > 0:
                buffer_obs_dim = self.replay_buffer.buffer[0][0].shape[0]
                if buffer_obs_dim != self.obs_size:
                    print(f"WARNING: Replay buffer observation size ({buffer_obs_dim}) "
                          f"does not match current agent obs_size ({self.obs_size}). "
                          "This can happen if resuming from an experiment with different observation specs. "
                          "Consider starting without --resume or clearing the buffer if issues occur.")

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
            'episode' : episode_num,
            'total_steps': self.total_steps,
            'actor' : self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1' : self.critic_1.state_dict(),
            'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
            'critic_2' : self.critic_2.state_dict(),
            'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
            'critic_target_1' : self.critic_target_1.state_dict(),
            'critic_target_2' : self.critic_target_2.state_dict(),
            'log_alpha' : self.log_alpha.detach().clone(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'replay_buffer_content' : list(self.replay_buffer.buffer) if self.arglist.save_replay_buffer else None
        }
        save_path = os.path.join(self.model_dir, "backup.ckpt")
        torch.save(checkpoint_state, save_path)
        print(f"Saved backup checkpoint to {save_path} at episode {episode_num}, total_steps {self.total_steps}")

    def soft_update(self, target_network, source_network, tau):
        with torch.no_grad():
            for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def log_video_to_wandb(self, episode_num, leg_lengths_dict, floor_friction_list):
        if not self.arglist.log_video and not self.arglist.log_video_local:
            return

        print(f"Generating video for episode {episode_num} (total steps {self.total_steps})")

        local_video_dir = os.path.join(self.exp_dir, "local_videos_exac")
        local_video_path = None
        if self.arglist.log_video_local:
            os.makedirs(local_video_dir, exist_ok=True)
            local_video_path = os.path.join(local_video_dir, f"episode_{episode_num}_steps_{self.total_steps}.gif")

        video_env = None
        collected_frames = []

        try:
            video_env = HalfCheetahMorphEnv(
                xml_path=self.xml_path,
                leg_lengths=leg_lengths_dict, # Env uses these for physics
                floor_friction=floor_friction_list,
                use_custom_morphology=True,
                render_mode=True
            )

            raw_obs = video_env.reset()
            # Augment observation for the actor
            current_augmented_obs = self._augment_observation(raw_obs, leg_lengths_dict)

            for _step in range(self.max_episode_steps):
                with torch.no_grad():
                    o_tensor = torch.tensor(current_augmented_obs, dtype=torch.float64, device=self.device).unsqueeze(0)
                    a_tensor, _ = self.actor(o_tensor, deterministic=True)
                action = a_tensor.cpu().numpy().squeeze()

                raw_next_obs, _reward, env_done, _info = video_env.step(action.astype(np.float32))
                # Augment next_obs for the actor if the loop continued (not strictly needed here as action is already taken)
                # but good for consistency if we were to store it or use it further.
                next_augmented_obs = self._augment_observation(raw_next_obs, leg_lengths_dict)
                current_augmented_obs = next_augmented_obs # For the next iteration if loop wasn't broken

                frame = video_env.render(mode='rgb_array')
                if frame is not None:
                    if _step < 5 or _step % 100 == 0 :
                         print(f"  Video Step {_step}: Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min():.2f}, max: {frame.max():.2f}")
                    collected_frames.append(frame.copy())
                else:
                    if _step < 5 or _step % 100 == 0 :
                        print(f"  Video Step {_step}: Frame is None!")

                if env_done:
                    break

            print(f"Collected {len(collected_frames)} frames for video in episode {episode_num}.")

            if collected_frames:
                video_np = np.array(collected_frames)
                print(f"  Final video_np shape: {video_np.shape}, dtype: {video_np.dtype}, min: {video_np.min():.2f}, max: {video_np.max():.2f}")

                if video_np.ndim != 4 or video_np.shape[0] == 0 or video_np.shape[3] != 3:
                    print(f"  Error: video_np has problematic shape: {video_np.shape}. Skipping save/log.")
                else:
                    if video_np.dtype != np.uint8:
                        print(f"  Converting video_np from {video_np.dtype} to uint8.")
                        if np.issubdtype(video_np.dtype, np.floating) and video_np.max() <= 1.0 and video_np.min() >= 0.0:
                            video_np = (video_np * 255).astype(np.uint8)
                        else:
                            video_np = np.clip(video_np, 0, 255).astype(np.uint8)
                        print(f"    After conversion: dtype: {video_np.dtype}, min: {video_np.min()}, max: {video_np.max()}")


                    if self.arglist.log_video_local and local_video_path:
                        try:
                            print(f"  Saving video locally to: {local_video_path}")
                            imageio.mimsave(local_video_path, video_np, fps=self.arglist.video_fps)
                            print(f"  Successfully saved video locally: {local_video_path}")
                        except Exception as e_local:
                            print(f"  Error saving video locally: {e_local}")

                    if self.arglist.log_video:
                        try:
                            print(f"  Attempting to log video to wandb...")
                            wandb.log({
                                "agent_rollout_video": wandb.Video(video_np, fps=self.arglist.video_fps, format="gif")
                            }, step=self.total_steps)
                            print(f"  Successfully attempted to log video to wandb for episode {episode_num}.")
                        except Exception as e_wandb:
                            print(f"  Error logging video to wandb: {e_wandb}")
            else:
                print(f"No frames collected for video (collected_frames is empty) at episode {episode_num}.")

        except Exception as e_outer:
            print(f"Error during video generation process (outer try-except): {e_outer}")
            import traceback
            traceback.print_exc()
        finally:
            if video_env:
                video_env.close()

    def train(self):
        run_name = f"seed_{self.arglist.seed}_{self.arglist.wandb_run_name_suffix}"
        wandb.init(
            project=self.arglist.wandb_project_name,
            name=run_name,
            config=vars(self.arglist),
            resume="allow",
            id=run_name if self.arglist.resume else None
        )
        # --- MODIFIED: Log to W&B config about observation change ---
        wandb.config.update({"observation_includes_leg_lengths": "manual_concat"}, allow_val_change=True)
        # --- END MODIFICATION ---

        print(f"Starting training for {self.arglist.episodes} episodes.")
        print(f"Logging to wandb project: {self.arglist.wandb_project_name}, run: {run_name}")
        if self.start_episode > 0:
             print(f"Resuming from episode {self.start_episode + 1}, total_steps {self.total_steps}")

        # Initial morphology (will be updated before first episode if start_episode is 0)
        current_leg_lengths = {key: 0.1 for key in self.morph_param_keys} # Placeholder
        current_floor_friction = [0.4, 0.1, 0.1] # Placeholder

        training_env = None

        for episode_idx in range(self.start_episode, self.arglist.episodes):
            current_episode_num = episode_idx + 1

            if episode_idx == self.start_episode or \
               (episode_idx > self.start_episode and episode_idx % self.arglist.morph_change_every == 0):
                current_leg_lengths = {
                    'b_thigh_len': random.uniform(0.0725, 0.725), 'b_shin_len': random.uniform(0.075, 0.75),
                    'f_thigh_len': random.uniform(0.0665, 0.665), 'f_shin_len': random.uniform(0.053, 0.53)
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

            if training_env is not None:
                training_env.close()

            try:
                training_env = HalfCheetahMorphEnv(
                    xml_path=self.xml_path,
                    leg_lengths=current_leg_lengths, # Env uses these for physics
                    floor_friction=current_floor_friction,
                    use_custom_morphology=True,
                    render_mode=False
                )
            except Exception as e:
                print(f"FATAL: Could not create training env at ep {current_episode_num}: {e}")
                wandb.finish(exit_code=1)
                return

            raw_obs_from_env = training_env.reset()
            current_augmented_obs = self._augment_observation(raw_obs_from_env, current_leg_lengths)
            episode_reward = 0
            episode_steps = 0

            for t in range(self.max_episode_steps):
                if self.total_steps < self.arglist.start_steps:
                    action = training_env.action_space.sample() # Raw action from env space
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(current_augmented_obs, dtype=torch.float64, device=self.device).unsqueeze(0)
                        action_tensor, _ = self.actor(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy().squeeze()

                raw_next_obs_from_env, reward, env_terminated, info = training_env.step(action.astype(np.float32))
                next_augmented_obs = self._augment_observation(raw_next_obs_from_env, current_leg_lengths)

                self.replay_buffer.push(current_augmented_obs, action, reward, next_augmented_obs)

                current_augmented_obs = next_augmented_obs # Update for the next iteration
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                if len(self.replay_buffer) >= self.arglist.replay_fill and \
                   self.total_steps % self.arglist.update_every == 0:

                    for _ in range(self.arglist.updates_per_step):
                        # obs_batch and next_obs_batch are already augmented from the buffer
                        obs_batch, action_batch, reward_batch, next_obs_batch = self.replay_buffer.sample(self.arglist.batch_size)

                        with torch.no_grad():
                            next_action_batch, next_log_prob_batch = self.actor(next_obs_batch, with_logprob=True)
                            q1_next_target = self.critic_target_1(next_obs_batch, next_action_batch)
                            q2_next_target = self.critic_target_2(next_obs_batch, next_action_batch)
                            min_q_next_target = torch.min(q1_next_target, q2_next_target)
                            alpha = torch.exp(self.log_alpha).detach()
                            target_q_value = reward_batch + self.arglist.gamma * (min_q_next_target - alpha * next_log_prob_batch)

                        q1_current = self.critic_1(obs_batch, action_batch)
                        q2_current = self.critic_2(obs_batch, action_batch)
                        critic_loss_1 = F.mse_loss(q1_current, target_q_value)
                        critic_loss_2 = F.mse_loss(q2_current, target_q_value)
                        critic_loss = critic_loss_1 + critic_loss_2

                        self.critic_optimizer_1.zero_grad()
                        critic_loss_1.backward()
                        self.critic_optimizer_1.step()

                        self.critic_optimizer_2.zero_grad()
                        critic_loss_2.backward()
                        self.critic_optimizer_2.step()

                        for p in self.critic_1.parameters(): p.requires_grad = False
                        for p in self.critic_2.parameters(): p.requires_grad = False

                        pi_action_batch, pi_log_prob_batch = self.actor(obs_batch, with_logprob=True)
                        q1_pi = self.critic_1(obs_batch, pi_action_batch)
                        q2_pi = self.critic_2(obs_batch, pi_action_batch)
                        min_q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = (alpha * pi_log_prob_batch - min_q_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        for p in self.critic_1.parameters(): p.requires_grad = True
                        for p in self.critic_2.parameters(): p.requires_grad = True

                        alpha_loss = -(self.log_alpha * (pi_log_prob_batch.detach() + self.target_entropy)).mean()

                        self.log_alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.log_alpha_optimizer.step()

                        alpha = torch.exp(self.log_alpha).detach()

                        self.soft_update(self.critic_target_1, self.critic_1, self.arglist.tau)
                        self.soft_update(self.critic_target_2, self.critic_2, self.arglist.tau)

                        if self.total_steps % self.arglist.log_interval == 0:
                            wandb.log({
                                "training/critic_loss": critic_loss.item(),
                                "training/actor_loss": actor_loss.item(),
                                "training/alpha_loss": alpha_loss.item(),
                                "training/alpha": alpha.item(),
                                "training/q1_current_mean": q1_current.mean().item(),
                                "training/q2_current_mean": q2_current.mean().item(),
                                "training/target_q_mean": target_q_value.mean().item(),
                                "training/log_prob_mean": pi_log_prob_batch.mean().item(),
                                "training/replay_buffer_size": len(self.replay_buffer),
                            }, step=self.total_steps)

                if env_terminated:
                    break

            print(f"Ep {current_episode_num}/{self.arglist.episodes} | Reward: {episode_reward:.2f} | Ep Steps: {episode_steps} | Total Steps: {self.total_steps}")
            wandb.log({
                "episode/reward": episode_reward,
                "episode/steps_count": episode_steps,
                "episode_num": current_episode_num,
            }, step=self.total_steps)

            if current_episode_num % self.arglist.save_model_every == 0 or current_episode_num == self.arglist.episodes:
                self.save_model_checkpoint(current_episode_num)

            if current_episode_num % self.arglist.backup_every == 0 or current_episode_num == self.arglist.episodes:
                if not (self.arglist.resume and episode_idx == self.start_episode and self.start_episode !=0):
                     self.save_backup_checkpoint(current_episode_num)

            if current_episode_num % self.arglist.video_log_every == 0:
                self.log_video_to_wandb(current_episode_num, current_leg_lengths, current_floor_friction)

        if training_env is not None:
            training_env.close()

        if self.arglist.log_video or self.arglist.log_video_local:
            self.log_video_to_wandb(self.arglist.episodes, current_leg_lengths, current_floor_friction)

        print("Training finished.")
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser("SAC with Custom HalfCheetah (Leg Lengths Manually Added to Obs), Random Morphology, and WandB Logging")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Execution mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU if available")
    # --- MODIFIED: log_dir for new experiment ---
    parser.add_argument("--log_dir", type=str, default="./logs_sac_morph_manual_obslegs", help="Base directory for logs and models")

    parser.add_argument("--episodes", type=int, default=5000, help="Total number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for actor, critics, and alpha")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training updates")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target networks")
    parser.add_argument("--initial_alpha", type=float, default=0.2, help="Initial value for temperature alpha")
    parser.add_argument("--start_steps", type=int, default=10000, help="Initial steps with random actions")
    parser.add_argument("--replay_size", type=int, default=int(1e6), help="Max replay buffer size")
    parser.add_argument("--replay_fill", type=int, default=10000, help="Min transitions before training starts")
    parser.add_argument("--update_every", type=int, default=1, help="Frequency (env steps) for training updates")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per training step")

    parser.add_argument("--morph_change_every", type=int, default=1, help="Frequency (episodes) to change morphology")

    parser.add_argument("--resume", action="store_true", default=False, help="Resume training")
    parser.add_argument("--save_model_every", type=int, default=250, help="Frequency (episodes) to save model checkpoint")
    parser.add_argument("--backup_every", type=int, default=100, help="Frequency (episodes) to save full backup checkpoint")
    parser.add_argument("--save_replay_buffer", action="store_true", default=False, help="Save replay buffer in backups")
    parser.add_argument("--log_interval", type=int, default=1000, help="Frequency (total_steps) to log training metrics")

    # --- MODIFIED: W&B Naming for New Experiment ---
    parser.add_argument("--wandb_project_name", type=str, default="SAC_HC_ManualObsLegs_exac", help="WandB project name")
    parser.add_argument("--wandb_run_name_suffix", type=str, default="manual_obslegs_exac", help="Suffix for WandB run name")
    # --- END MODIFICATION ---

    parser.add_argument("--log_video", action="store_true", default=True, help="Log videos of agent rollouts to WandB")
    parser.add_argument("--log_video_local", action="store_true", default=False, help="Log videos locally to a file in the experiment directory")
    parser.add_argument("--video_log_every", type=int, default=10, help="Frequency (in episodes) to log videos")
    parser.add_argument("--video_fps", type=int, default=30, help="FPS for logged videos")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    sac_agent = SAC(args)