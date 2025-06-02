import os
import argparse
from copy import deepcopy
from collections import deque
import math
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.distributions.normal import Normal
import wandb
# from dm_control import suite # Removed dm_control suite import
# import glob # Removed glob as video saving is commented out
# import subprocess # Removed subprocess as video saving is commented out
# import imageio # Removed imageio as GIF saving is commented out

# Import your custom environment
from halfcheetah_morph_env import HalfCheetahMorphEnv

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, o, a, r, o_1):
         self.buffer.append((o, a, r, o_1)) # Store as numpy arrays

    def sample(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        # Convert to tensors here during sampling
        return torch.tensor(np.array(O), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float64, device=self.device).unsqueeze(1),\
               torch.tensor(np.array(O_1), dtype=torch.float64, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Critic network
class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)
        return y

# Actor network
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y2)
            log_sigma = torch.clamp(log_sigma,min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample() # Sample with reparameterization trick
            if with_logprob:
                # calculate log_prob of the squashed action
                log_prob = dist.log_prob(x_t)
                # Correction term for tanh squashing
                log_prob -= torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6) # Added small epsilon for stability
                log_prob = log_prob.sum(dim=-1, keepdim=True) # Sum over action dimensions
            else:
                log_prob = None
            action = torch.tanh(x_t) # Squish to [-1, 1]

        # For HalfCheetah, action space is [-1, 1], so tanh is sufficient.
        # If the action space were different, we'd scale/offset here.

        return action, log_prob


# Soft Actor-Critic algorithm
class SAC:
    def __init__(self, arglist):
        self.arglist = arglist

        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)

        # --- Custom Environment Setup ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "half_cheetah.xml") # Assuming XML is in the same directory

        # Specific morphology parameters as requested
        fixed_leg_lengths = dict(
            b_thigh_len=0.8,
            b_shin_len=0.4,
            f_thigh_len=0.6,
            f_shin_len=0.3
        )
        fixed_floor_friction = [0.5, 0.5, 0.5] # [friction_x, friction_y, torsion]

        # Determine render_mode based on requested mode and flags
        # should_render = (self.arglist.mode == "eval" and (self.arglist.render or self.arglist.save_video))
        should_render = False # Set render_mode to False for training

        self.env = HalfCheetahMorphEnv(
            xml_path=xml_path,
            leg_lengths=fixed_leg_lengths,
            floor_friction=fixed_floor_friction,
            use_custom_morphology=False,  # Use the specified custom morphology
            render_mode=should_render
        )

        # Get space sizes from the custom environment
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        # --- End Custom Environment Setup ---

        # --- Episode Step Limit ---
        # Define max steps per episode since environment done flag is always False
        self.max_episode_steps = 1000 # Set a reasonable limit like 1000 steps
        # --- End Episode Step Limit ---


        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        self.actor = Pi_FC(self.obs_size,self.action_size).to(self.device)

        if self.arglist.mode == "train":
            self.critic_1 = Q_FC(self.obs_size,self.action_size).to(self.device)
            self.critic_target_1 = deepcopy(self.critic_1)
            self.critic_loss_fn_1 =  torch.nn.MSELoss()

            self.critic_2 = Q_FC(self.obs_size,self.action_size).to(self.device)
            self.critic_target_2 = deepcopy(self.critic_2)
            self.critic_loss_fn_2 =  torch.nn.MSELoss()

            # set target entropy to -|A|
            self.target_entropy = - self.action_size

            # Use a fixed name for the log directory specific to this setup
            path = "./log/halfcheetah_morph_fixed" # Using a fixed name
            self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
            self.model_dir = os.path.join(self.exp_dir, "models")
            self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

            if self.arglist.resume:
                try:
                    checkpoint = torch.load(os.path.join(self.model_dir,"backup.ckpt"), map_location=self.device)
                    self.start_episode = checkpoint['episode'] + 1

                    self.actor.load_state_dict(checkpoint['actor'])
                    self.critic_1.load_state_dict(checkpoint['critic_1'])
                    self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
                    self.critic_2.load_state_dict(checkpoint['critic_2'])
                    self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
                    # Need to re-create log_alpha tensor with requires_grad=True
                    self.log_alpha = torch.tensor(checkpoint['log_alpha'].item(), dtype=torch.float64, device=self.device, requires_grad=True)


                    # Recreate replay buffer and load state
                    self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
                    # Load buffer content from saved list
                    for item in checkpoint['replay_buffer_content']:
                        self.replay_buffer.push(*item) # Push items back one by one


                    print(f"Done loading checkpoint from episode {checkpoint['episode']}")
                    print(f"Replay buffer size after loading: {len(self.replay_buffer)}")


                except FileNotFoundError:
                    print("Checkpoint file not found. Starting training from scratch.")
                    self.start_episode = 0
                    self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)
                    self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                    self.start_episode = 0
                    self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)
                    self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)


            else: # Not resuming
                self.start_episode = 0
                self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)
                self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

                # Create directories if they don't exist
                os.makedirs(self.model_dir, exist_ok=True)
                os.makedirs(self.tensorboard_dir, exist_ok=True)


            for param in self.critic_target_1.parameters():
                param.requires_grad = False

            for param in self.critic_target_2.parameters():
                param.requires_grad = False

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.arglist.lr)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.arglist.lr)

            if self.arglist.resume:
                 try:
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1'])
                    self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])
                    self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
                    print("Done loading optimizer states.")
                 except Exception as e:
                     print(f"Error loading optimizer states: {e}. Starting optimizer training from scratch.")


            self.train()

        # elif self.arglist.mode == "eval":
        #     try:
        #         checkpoint = torch.load(self.arglist.checkpoint, map_location=self.device)
        #         self.actor.load_state_dict(checkpoint['actor'])
        #         print(f"Loaded model from {self.arglist.checkpoint}")
        #         # Pass render/save_video flags directly to eval
        #         self.eval(self.arglist.episodes, self.arglist.render, self.arglist.save_video)
        #     except FileNotFoundError:
        #          print(f"Error: Checkpoint file not found at {self.arglist.checkpoint}")
        #     except Exception as e:
        #          print(f"Error loading model or during evaluation: {e}")



    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict()}
        save_path = os.path.join(self.model_dir, name)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def save_backup(self, episode):
        checkpoint = {'episode' : episode,\
                      'actor' : self.actor.state_dict(),\
                      'actor_optimizer': self.actor_optimizer.state_dict(),\
                      'critic_1' : self.critic_1.state_dict(),\
                      'critic_optimizer_1': self.critic_optimizer_1.state_dict(),\
                      'critic_2' : self.critic_2.state_dict(),\
                      'critic_optimizer_2': self.critic_optimizer_2.state_dict(),\
                      'critic_target_1' : self.critic_target_1.state_dict(),\
                      'critic_target_2' : self.critic_target_2.state_dict(),\
                      'log_alpha' : self.log_alpha.detach(),\
                      'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),\
                      'replay_buffer_content' : list(self.replay_buffer.buffer) # Save buffer content as a list
                      }
        save_path = os.path.join(self.model_dir, "backup.ckpt")
        torch.save(checkpoint, save_path)
        print(f"Saved backup checkpoint to {save_path} at episode {episode}")


    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        # Initialize wandb run
        wandb.init(
            project="HalfCheetahMorph_SAC",  # Change to your preferred project name
            name=f"seed_{self.arglist.seed}_run",  # Unique run name
            config=vars(self.arglist),  # Log hyperparameters
        )
        print(f"Starting training for {self.arglist.episodes} episodes, max {self.max_episode_steps} steps per episode...")
        print(f"Starting from episode {self.start_episode}")

        total_steps = 0 # Keep track of total environment steps

        for episode in range(self.start_episode, self.arglist.episodes):
            # Reset environment at the beginning of each episode
            o = self.env.reset()
            ep_r = 0
            step_count = 0
            # The environment's done flag is always False, so we use step_count
            episode_done = False # Flag to track if the episode (defined by step limit) is over

            # Run for max_episode_steps or until env signals done (if it ever did)
            while step_count < self.max_episode_steps and not episode_done:
                if self.replay_buffer.__len__() >= self.arglist.start_steps:
                    # Agent takes action based on policy after sufficient exploration
                    with torch.no_grad():
                        # Ensure observation is a tensor and unsqueezed for batch dim
                        o_tensor = torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0)
                        a_tensor, _ = self.actor(o_tensor) # Use stochastic action for training
                    a = a_tensor.cpu().numpy()[0]
                else:
                    # Random action for initial exploration
                    a = self.env.action_space.sample()

                # Use env.step() directly - it returns obs, reward, done, info
                o_1, r, env_done, info = self.env.step(a.astype(np.float32)) # Ensure action is float32 for env

                # In this specific environment, env_done is always False.
                # We define episode end by max_episode_steps.
                # However, if the environment *could* signal done, we would respect it.
                episode_done = env_done # Use the env's done flag, although it's always False here

                self.replay_buffer.push(o, a, r, o_1) # Store transition

                ep_r += r # Accumulate reward for the episode
                o = o_1 # Update observation
                step_count += 1
                total_steps += 1 # Increment total steps

                # Only start training after replay buffer has enough elements
                if len(self.replay_buffer) >= self.arglist.replay_fill:
                    O, A, R_tensor, O_1 = self.replay_buffer.sample(self.arglist.batch_size)

                    # --- Critic Update ---
                    with torch.no_grad():
                        # Target actions come from *current* policy
                        A_1, logp_A_1 = self.actor(O_1, deterministic=False, with_logprob=True)

                        # Compute twin Q-values for next state-action pair
                        next_q_value_1 = self.critic_target_1(O_1, A_1)
                        next_q_value_2 = self.critic_target_2(O_1, A_1)
                        next_q_value = torch.min(next_q_value_1, next_q_value_2)

                        # Compute target Q value using Bellman equation
                        # The reward R is a tensor of shape (batch_size, 1) from sample()
                        target_q_value = R_tensor + self.arglist.gamma * (next_q_value - torch.exp(self.log_alpha).detach() * logp_A_1)
                        # target_q_value = target_q_value.detach() # Target is already detached implicitly by torch.no_grad()

                    # Compute current Q values
                    q_value_1 = self.critic_1(O, A)
                    q_value_2 = self.critic_2(O, A)

                    # Compute critic loss (MSE against target Q)
                    critic_loss_1 = self.critic_loss_fn_1(q_value_1, target_q_value)
                    critic_loss_2 = self.critic_loss_fn_2(q_value_2, target_q_value)
                    critic_loss = critic_loss_1 + critic_loss_2

                    # Optimize critics
                    self.critic_optimizer_1.zero_grad()
                    critic_loss_1.backward()
                    self.critic_optimizer_1.step()

                    self.critic_optimizer_2.zero_grad()
                    critic_loss_2.backward()
                    self.critic_optimizer_2.step()

                    # --- Actor and Alpha Update (Delayed Update) ---
                    # Freeze critics when updating actor
                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                         param_1.requires_grad = False
                         param_2.requires_grad = False

                    # Compute actor loss
                    # Sample actions from current policy for observed states O
                    A_pi, logp_A_pi = self.actor(O, deterministic=False, with_logprob=True)
                    # Get Q values for these actions
                    q_value_pi_1 = self.critic_1(O, A_pi)
                    q_value_pi_2 = self.critic_2(O, A_pi)
                    q_value_pi = torch.min(q_value_pi_1, q_value_pi_2)

                    # Actor loss: maximize Q value minus temperature-weighted entropy
                    actor_loss = -torch.mean(q_value_pi - torch.exp(self.log_alpha).detach() * logp_A_pi)

                    # Optimize actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Unfreeze critics
                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                        param_1.requires_grad = True
                        param_2.requires_grad = True

                    # --- Update Temperature (Alpha) ---
                    # Compute alpha loss
                    # Use logp from the actor update step
                    alpha_loss = (torch.exp(self.log_alpha) * (-logp_A_pi.detach() - self.target_entropy)).mean() # Use detached logp

                    # Optimize alpha
                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

                    # --- Soft Target Update ---
                    self.soft_update(self.critic_target_1, self.critic_1, self.arglist.tau)
                    self.soft_update(self.critic_target_2, self.critic_2, self.arglist.tau)
                    # --- Logging ---
                    if total_steps % 1000 == 0:
                        wandb.log({
                            "critic_loss": critic_loss.item(),
                            "actor_loss": actor_loss.item(),
                            "alpha_loss": alpha_loss.item(),
                            "q_value_mean": q_value_pi.mean().item(),
                            "logp_mean": logp_A_pi.mean().item()
                        })


            # End of episode (step limit reached or env_done is True)
            print(f"Episode {episode+1}/{self.arglist.episodes}: Total Reward = {ep_r:.4f}, Steps = {step_count}")
            # Log total episode reward at the end of the episode

            wandb.log({
                "episode": episode,
                "episode_reward": ep_r,
                "alpha": torch.exp(self.log_alpha).item(),
                "steps": total_steps
            })



            # Evaluate agent performance periodically - COMMENTED OUT
            # if (episode + 1) % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
            #      eval_ep_r_list = self.eval(self.arglist.eval_over, render=False, save_video=False)
            #      avg_eval_r = np.mean(eval_ep_r_list)
            #      self.save_checkpoint(f"{episode+1}.ckpt") # Save checkpoint after eval

            # Save checkpoint at the end of each evaluation interval (or last episode)
            # Since eval is commented out, we'll save a checkpoint here based on the training episode count
            if (episode + 1) % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                 self.save_checkpoint(f"{episode+1}.ckpt") # Save checkpoint periodically during train


            # Save backup checkpoint periodically
            if (episode + 1) % 250 == 0 or episode == self.arglist.episodes-1: # Save every 250 episodes + last episode
                 if (episode + 1) > self.start_episode: # Only save if we've done at least one new episode
                    self.save_backup(episode)

        print("Training finished.")
        wandb.finish()
        self.env.close() # Close the environment after training

    # Evaluation method - COMMENTED OUT
    # def eval(self, episodes, render=False, save_video=False):
    #     # Evaluate agent performance over several episodes
    #     print(f"Starting evaluation for {episodes} episodes, max {self.max_episode_steps} steps per episode (render={render}, save_video={save_video})...")

    #     ep_r_list = []
    #     # Removed video related lists and logic

    #     for episode in range(episodes):
    #         # Reset environment for each evaluation episode
    #         o = self.env.reset()
    #         ep_r = 0
    #         step_count = 0
    #         episode_done = False

    #         while step_count < self.max_episode_steps and not episode_done:
    #             with torch.no_grad():
    #                 o_tensor = torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0)
    #                 # Use deterministic=True for evaluation policy
    #                 a_tensor, _ = self.actor(o_tensor, deterministic=True)
    #             a = a_tensor.cpu().numpy()[0] # Action is already in [-1, 1] range from tanh

    #             # Use env.step()
    #             o_1, r, env_done, info = self.env.step(a.astype(np.float32)) # Ensure action is float32

    #             episode_done = env_done

    #             ep_r += r # Accumulate reward
    #             o = o_1
    #             step_count += 1

    #             # Removed render calls

    #         # Episode finished (step limit reached or env_done True)
    #         ep_r_list.append(ep_r)
    #         print(f"  Eval Episode {episode+1}/{episodes}: Total Reward = {ep_r:.4f}, Steps = {step_count}")

    #     avg_return = np.mean(ep_r_list)
    #     print(f"Evaluation complete. Average return over {episodes} episodes: {avg_return:.4f}")

    #     # Removed video saving logic

    #     return ep_r_list

def parse_args():
    parser = argparse.ArgumentParser("SAC with Custom HalfCheetah")
    # Common settings
    parser.add_argument("--mode", type=str, default="train", help="train or eval") # Default to train
    parser.add_argument("--episodes", type=int, default=10000, help="number of episodes (for train or eval)") # Default episodes
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("--lr", type=float, default=3e-4, help="actor, critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--start-steps", type=int, default=int(1e4), help="steps for random action exploration at start")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e4), help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes during training") # This is now used for saving checkpoints instead of running eval
    parser.add_argument("--eval-over", type=int, default=5, help="each time eval over _ episodes") # This arg is now unused
    # Eval settings (These args are still parsed but will not trigger evaluation logic)
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint for evaluation") # Unused
    parser.add_argument("--render", action="store_true", default=False, help="render evaluation episodes") # Unused
    parser.add_argument("--save-video", action="store_true", default=False, help="save video of evaluation episodes (requires --render)") # Unused

    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    # The morphology parameters (leg_lengths, friction) are hardcoded
    # within the SAC class for this specific implementation based on your request.

    sac = SAC(arglist)