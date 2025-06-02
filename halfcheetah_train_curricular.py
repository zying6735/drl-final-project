# halfcheetah_curriculum_train.py

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
import time # Import time for unique run naming
import mujoco # Import mujoco to get default parameters from XML
import xml.etree.ElementTree as ET # Import for parsing XML

# Import your custom environment and XML modification functions
from halfcheetah_morph_env import HalfCheetahMorphEnv

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, o, a, r, o_1):
         # Store as numpy arrays
         self.buffer.append((o, a, r, o_1))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Replay buffer does not contain enough elements for sampling.")
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        # Convert to tensors here during sampling
        return torch.tensor(np.array(O), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float64, device=self.device).unsqueeze(1),\
               torch.tensor(np.array(O_1), dtype=torch.float64, device=self.device)

    def __len__(self):
        return len(self.buffer)

    def get_content(self):
        # Return buffer content as a list for saving
        return list(self.buffer)

    def load_content(self, content):
        # Load content into the buffer
        self.buffer = deque(content, maxlen=self.buffer.maxlen)


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


# Soft Actor-Critic algorithm Agent
class SACAgent:
    def __init__(self, obs_size, action_size, arglist):
        self.arglist = arglist
        self.obs_size = obs_size
        self.action_size = action_size

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        self.actor = Pi_FC(self.obs_size,self.action_size).to(self.device)

        self.critic_1 = Q_FC(self.obs_size,self.action_size).to(self.device)
        self.critic_target_1 = deepcopy(self.critic_1)
        self.critic_loss_fn_1 =  torch.nn.MSELoss()

        self.critic_2 = Q_FC(self.obs_size,self.action_size).to(self.device)
        self.critic_target_2 = deepcopy(self.critic_2)
        self.critic_loss_fn_2 =  torch.nn.MSELoss()

        # set target entropy to -|A|
        self.target_entropy = - self.action_size

        # Alpha (temperature parameter)
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.arglist.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.arglist.lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

        # Freeze target critics initially
        for param in self.critic_target_1.parameters():
            param.requires_grad = False
        for param in self.critic_target_2.parameters():
            param.requires_grad = False

        # Define max steps per episode since environment done flag is always False
        self.max_episode_steps = 1000 # Set a reasonable limit like 1000 steps


    def save_state(self, path, total_episodes_trained, total_steps_trained):
        """Saves the agent's state including networks, optimizers, alpha, replay buffer, and training progress."""
        checkpoint = {
            'total_episodes_trained' : total_episodes_trained,
            'total_steps_trained' : total_steps_trained,
            'actor' : self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1' : self.critic_1.state_dict(),
            'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
            'critic_2' : self.critic_2.state_dict(),
            'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
            'critic_target_1' : self.critic_target_1.state_dict(),
            'critic_target_2' : self.critic_target_2.state_dict(),
            'log_alpha' : self.log_alpha.detach().cpu().item(), # Save as item
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'replay_buffer_content' : self.replay_buffer.get_content() # Save buffer content
        }
        torch.save(checkpoint, path)
        print(f"Saved agent state to {path}")

    def load_state(self, path):
        """Loads the agent's state from a checkpoint file."""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor'])
            self.critic_1.load_state_dict(checkpoint['critic_1'])
            self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
            self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
            self.critic_2.load_state_dict(checkpoint['critic_2'])


            # Need to re-create log_alpha tensor with requires_grad=True
            self.log_alpha = torch.tensor(checkpoint['log_alpha'], dtype=torch.float64, device=self.device, requires_grad=True)

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1'])
            self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])

            # Load replay buffer content
            self.replay_buffer.load_content(checkpoint['replay_buffer_content'])

            total_episodes_trained = checkpoint.get('total_episodes_trained', 0) # Get episode count, default to 0 if not found
            total_steps_trained = checkpoint.get('total_steps_trained', 0) # Get step count, default to 0 if not found

            print(f"Loaded agent state from {path}.")
            print(f"Total episodes trained before loading: {total_episodes_trained}")
            print(f"Total steps trained before loading: {total_steps_trained}")
            print(f"Replay buffer size after loading: {len(self.replay_buffer)}")
            return total_episodes_trained, total_steps_trained

        except FileNotFoundError:
            print(f"Checkpoint file not found at {path}. Starting from scratch.")
            return 0, 0
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}. Starting from scratch.")
            # Re-initialize components if loading failed
            self.__init__(self.obs_size, self.action_size, self.arglist)
            return 0, 0


    def soft_update(self, target, source, tau):
        """Performs a soft update of the target network parameters."""
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def run_stage(self, xml_path, stage_config, total_steps_offset, total_episodes_offset):
        """
        Trains the agent for a specific curriculum stage.

        Args:
            xml_path (str): Path to the base MuJoCo XML file.
            stage_config (dict): Configuration for the current stage (episodes, morphology/friction rules).
            total_steps_offset (int): The total number of environment steps completed before this stage.
            total_episodes_offset (int): The total number of episodes completed before this stage.

        Returns:
            tuple: (steps_taken_in_stage, episodes_completed_in_stage)
        """
        stage_episodes = stage_config['episodes_per_stage']
        morphology_rule = stage_config['morphology_rule']
        friction_rule = stage_config['friction_rule']
        morphology_params = stage_config.get('morphology_params', {}) # Ranges or fixed values
        friction_params = stage_config.get('friction_params', {}) # Ranges or fixed values

        print(f"--- Starting Stage: {stage_config['name']} ---")
        print(f"Training for {stage_episodes} episodes.")
        print(f"Morphology Rule: {morphology_rule}")
        print(f"Friction Rule: {friction_rule}")
        print(f"Morphology Params: {morphology_params}")
        print(f"Friction Params: {friction_params}")
        print("-" * 30)


        steps_taken_in_stage = 0

        for episode_in_stage in range(stage_episodes):
            current_episode_global = total_episodes_offset + episode_in_stage

            # --- Determine Environment Parameters for this Episode ---
            leg_lengths = None
            floor_friction = None # This will be a 3-element list [slide, roll, torsion]

            if morphology_rule == 'fixed':
                leg_lengths = morphology_params['fixed_values']
            elif morphology_rule == 'random':
                 # Sample random morphology within the specified ranges
                 leg_lengths = dict(
                    b_thigh_len=random.uniform(morphology_params['b_thigh_len'][0], morphology_params['b_thigh_len'][1]),
                    b_shin_len=random.uniform(morphology_params['b_shin_len'][0], morphology_params['b_shin_len'][1]),
                    f_thigh_len=random.uniform(morphology_params['f_thigh_len'][0], morphology_params['f_thigh_len'][1]),
                    f_shin_len=random.uniform(morphology_params['f_shin_len'][0], morphology_params['f_shin_len'][1])
                 )

            if friction_rule == 'fixed':
                # floor_friction is a 3-element vector, we only set the first element (slide)
                floor_friction = [friction_params['fixed_value'], 0.1, 0.1] # Assuming 0.1 for roll/torsion as per original code
            elif friction_rule == 'random':
                 # Sample random friction for the first element (slide)
                 friction_value = random.uniform(friction_params['range'][0], friction_params['range'][1])
                 floor_friction = [friction_value, 0.1, 0.1] # Assuming 0.1 for roll/torsion


            # --- Create Environment for this Episode ---
            # Recreate environment for each episode to apply morphology/friction changes
            try:
                env = HalfCheetahMorphEnv(
                    xml_path=xml_path,
                    leg_lengths=leg_lengths,
                    floor_friction=floor_friction,
                    use_custom_morphology=True, # Always use custom morphology based on rules
                    render_mode=False # No rendering during training
                )
                o = env.reset()
            except RuntimeError as e:
                print(f"Error creating environment for episode {current_episode_global}: {e}. Skipping episode.")
                # Log error or handle appropriately, maybe skip to next episode
                wandb.log({"env_creation_error": str(e)}, step=total_steps_offset + steps_taken_in_stage)
                continue # Skip to the next episode


            ep_r = 0
            step_count_in_episode = 0
            # episode_done = False # This env's done is always False, we use max_episode_steps

            # Run for max_episode_steps
            while step_count_in_episode < self.max_episode_steps: # and not episode_done: # env_done is always False
                current_step_global = total_steps_offset + steps_taken_in_stage

                if len(self.replay_buffer) >= self.arglist.start_steps:
                    # Agent takes action based on policy after sufficient exploration
                    with torch.no_grad():
                        # Ensure observation is a tensor and unsqueezed for batch dim
                        o_tensor = torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0)
                        a_tensor, _ = self.actor(o_tensor) # Use stochastic action for training
                    a = a_tensor.cpu().numpy()[0]
                else:
                    # Random action for initial exploration
                    a = env.action_space.sample()

                # Use env.step() directly - it returns obs, reward, done, info
                # Ensure action is float32 for env
                o_1, r, env_done_flag, info = env.step(a.astype(np.float32))

                # In this specific environment, env_done_flag is always False.
                # We define episode end by max_episode_steps.
                # episode_done = env_done_flag # Use the env's done flag if it were meaningful

                self.replay_buffer.push(o, a, r, o_1) # Store transition

                ep_r += r # Accumulate reward for the episode
                o = o_1 # Update observation
                step_count_in_episode += 1
                steps_taken_in_stage += 1 # Increment total steps for the stage

                # Only start training after replay buffer has enough elements
                if len(self.replay_buffer) >= self.arglist.replay_fill:
                    # Perform SAC updates
                    try:
                        O, A, R_tensor, O_1 = self.replay_buffer.sample(self.arglist.batch_size)
                    except ValueError:
                         # Not enough samples yet, skip training step
                         continue

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

                    # --- Logging Training Metrics ---
                    if current_step_global % 1000 == 0:
                         wandb.log({
                            f"stage_{stage_config['id']}/critic_loss": critic_loss.item(),
                            f"stage_{stage_config['id']}/actor_loss": actor_loss.item(),
                            f"stage_{stage_config['id']}/alpha_loss": alpha_loss.item(),
                            f"stage_{stage_config['id']}/q_value_mean": q_value_pi.mean().item(),
                            f"stage_{stage_config['id']}/logp_mean": logp_A_pi.mean().item(),
                            f"stage_{stage_config['id']}/alpha": torch.exp(self.log_alpha).item(),
                            "global_step": current_step_global
                         })


            # --- End of Episode ---
            env.close() # Close environment after each episode

            print(f"""Stage {stage_config['id']} ({stage_config['name']}) | Episode {episode_in_stage+1}/{stage_episodes} (Global: {current_episode_global+1})
                    Total Reward = {ep_r:.4f}, Steps = {step_count_in_episode}
                    Morphology: {leg_lengths}
                    Friction: {floor_friction}
                    Replay Buffer Size: {len(self.replay_buffer)}
                    -------------------------------------------
                    """)

            # Log episode metrics
            wandb.log({
                f"stage_{stage_config['id']}/episode_reward": ep_r,
                f"stage_{stage_config['id']}/episode_steps": step_count_in_episode,
                "global_episode": current_episode_global + 1,
                "global_step": total_steps_offset + steps_taken_in_stage, # Log total steps after episode
            })

            # Save backup checkpoint periodically within the stage
            if (episode_in_stage + 1) % self.arglist.save_every_stage_episodes == 0 or (episode_in_stage + 1) == stage_episodes:
                 # Define a path specific to the stage and episode
                 backup_path = os.path.join(self.arglist.log_dir, f"stage_{stage_config['id']}_ep_{episode_in_stage+1}_backup.ckpt")
                 self.save_state(backup_path, current_episode_global + 1, total_steps_offset + steps_taken_in_stage)

            # Save model checkpoint every 100 episodes within the stage
            if (episode_in_stage + 1) % 100 == 0:
                model_ckpt_path = os.path.join(self.arglist.log_dir, f"stage_{stage_config['id']}_ep_{episode_in_stage+1}.ckpt")
                self.save_state(model_ckpt_path, current_episode_global + 1, total_steps_offset + steps_taken_in_stage)



        print(f"--- Finished Stage: {stage_config['name']} ---")
        return steps_taken_in_stage, stage_episodes # Return actual steps taken and episodes completed



def get_default_env_params(xml_path):
    return {
        'b_thigh_len': 0.2,
        'b_shin_len': 0.2,
        'f_thigh_len': 0.14,
        'f_shin_len': 0.13
    }, 0.4


def parse_args():
    parser = argparse.ArgumentParser("SAC Curriculum Training with Custom HalfCheetah")
    # Common settings
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--log-dir", type=str, default="./log/halfcheetah_curriculum", help="directory for logs and checkpoints")
    # Core training parameters (passed to SACAgent)
    parser.add_argument("--lr", type=float, default=3e-4, help="actor, critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--start-steps", type=int, default=int(1e4), help="steps for random action exploration at start")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e4), help="elements in replay buffer before training starts")
    parser.add_argument("--save-every-stage-episodes", type=int, default=50, help="save backup checkpoint every _ episodes within a stage")
    parser.add_argument("--episodes-per-stage", type=int, default=1000, help="number of episodes to train for in each curriculum stage")
    parser.add_argument("--resume_stage", type=int, default=1, help="stage to resume from (1-4)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="path to checkpoint file to resume from")


    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()

    # Set seeds
    random.seed(arglist.seed)
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arglist.seed)

    # Create log directory
    os.makedirs(arglist.log_dir, exist_ok=True)

    # Initialize wandb run
    # Use a unique name based on time and seed
    run_name = f"curriculum_seed_{arglist.seed}_{int(time.time())}"
    wandb.init(
        project="HalfCheetahCurriculum_SAC", # Your project name
        name=run_name,
        config=vars(arglist),
    )

    # --- Get Default Environment Parameters from XML ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "half_cheetah.xml")
    default_leg_lengths, default_floor_friction_slide = get_default_env_params(xml_path)
    print(f"Default Leg Lengths (from XML): {default_leg_lengths}")
    print(f"Default Floor Friction (slide, from XML): {default_floor_friction_slide}")

    # Define morphology and friction ranges based on user specs
    MORPHOLOGY_RANGES = dict(
        b_thigh_len=[0.0725, 0.725],
        b_shin_len=[0.075, 0.75],
        f_thigh_len=[0.0675, 0.675],
        f_shin_len=[0.053, 0.53]
    )

    FRICTION_RANGE = [0.2, 2.0] # User specified 0.2-2.0 for random tasks

    # --- Define the curriculum stages using default and specified values ---
    CURRICULUM_STAGES = [
        {
            'id': 1,
            'name': 'Task 1 (Base)',
            'episodes_per_stage': arglist.episodes_per_stage,
            'morphology_rule': 'fixed',
            'morphology_params': {'fixed_values': default_leg_lengths}, # Use default
            'friction_rule': 'fixed',
            'friction_params': {'fixed_value': default_floor_friction_slide} # Use default
        },
        {
            'id': 2,
            'name': 'Task 2 (Random Morphology)',
            'episodes_per_stage': arglist.episodes_per_stage,
            'morphology_rule': 'random',
            'morphology_params': MORPHOLOGY_RANGES, # Use the full ranges
            'friction_rule': 'fixed',
            'friction_params': {'fixed_value': default_floor_friction_slide} # Use default
        },
        {
            'id': 3,
            'name': 'Task 3 (Random Friction)',
            'episodes_per_stage': arglist.episodes_per_stage,
            'morphology_rule': 'fixed',
            'morphology_params': {'fixed_values': default_leg_lengths}, # Use default
            'friction_rule': 'random',
            'friction_params': {'range': FRICTION_RANGE} # Use the random range [0.2, 2.0]
        },
        {
            'id': 4,
            'name': 'Task 4 (Random All)',
            'episodes_per_stage': arglist.episodes_per_stage,
            'morphology_rule': 'random',
            'morphology_params': MORPHOLOGY_RANGES, # Use the full ranges
            'friction_rule': 'random',
            'friction_params': {'range': FRICTION_RANGE} # Use the random range [0.2, 2.0]
        }
    ]


    # --- Get Observation and Action Sizes ---
    # Create a dummy environment just to get the sizes
    try:
        dummy_env = HalfCheetahMorphEnv(xml_path=xml_path, use_custom_morphology=False) # Use default morphology for size check
        obs_size = dummy_env.observation_space.shape[0]
        action_size = dummy_env.action_space.shape[0]
        dummy_env.close()
        print(f"Observation size: {obs_size}, Action size: {action_size}")
    except Exception as e:
        print(f"Error creating dummy environment to get sizes: {e}")
        exit() # Exit if we can't even get the environment sizes


    # --- Curriculum Training Loop ---
    agent = SACAgent(obs_size, action_size, arglist) # Initialize the agent once
    total_steps_offset = 0
    total_episodes_offset = 0
    last_stage_save_path = arglist.resume_checkpoint # Use resume_checkpoint if provided

    # Find the starting stage index
    start_stage_index = -1
    for i, stage in enumerate(CURRICULUM_STAGES):
        if stage['id'] == arglist.resume_stage:
            start_stage_index = i
            break

    if start_stage_index == -1:
        print(f"Error: Resume stage {arglist.resume_stage} not found in curriculum.")
        exit()

    # Load state if resuming
    if last_stage_save_path:
        loaded_episodes, loaded_steps = agent.load_state(last_stage_save_path)
        total_episodes_offset = loaded_episodes
        total_steps_offset = loaded_steps
        print(f"Resuming training from Stage {arglist.resume_stage}, Global Episode {total_episodes_offset}, Global Step {total_steps_offset}")
    else:
        print(f"Starting training from Stage {arglist.resume_stage} (from scratch).")


    for stage_index in range(start_stage_index, len(CURRICULUM_STAGES)):
        stage_config = CURRICULUM_STAGES[stage_index]
        stage_id = stage_config['id']

        print(f"\n{'='*30}\nStarting Stage {stage_id}: {stage_config['name']}\n{'='*30}")

        # Run training for the current stage
        steps_in_stage, episodes_in_stage = agent.run_stage(
            xml_path=xml_path,
            stage_config=stage_config,
            total_steps_offset=total_steps_offset,
            total_episodes_offset=total_episodes_offset
        )

        # Update offsets for the next stage
        total_steps_offset += steps_in_stage
        total_episodes_offset += episodes_in_stage

        # Save the agent's state at the end of the stage
        # This checkpoint will be used as the starting point for the next stage
        current_stage_save_path = os.path.join(arglist.log_dir, f"stage_{stage_id}_final.ckpt")
        agent.save_state(current_stage_save_path, total_episodes_offset, total_steps_offset)
        last_stage_save_path = current_stage_save_path # Set for the next iteration

        print(f"\nFinished Stage {stage_id}. Total episodes trained: {total_episodes_offset}, Total steps: {total_steps_offset}")


    print("\nCurriculum training finished.")
    wandb.finish()