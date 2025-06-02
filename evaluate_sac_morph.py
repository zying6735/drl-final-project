# HOW TO USE THIS SCRIPT:
# 1. Ensure this script (`evaluate_sac_morph.py`) is in the same directory as:
#    - Your SAC training script (e.g., `sac_train_script.py` - containing `Pi_FC` definition).
#    - `halfcheetah_morph_env.py` (your custom environment file).
#    - `half_cheetah.xml` (the MuJoCo model file).
#
# 2. Modify the import for `Pi_FC` if your training script has a different name:
#    Near the top, find `from sac_train_script import Pi_FC` and change `sac_train_script`
#    to the actual filename of your training script (without the .py extension).
#
# 3. Open your terminal or Anaconda Prompt.
#
# 4. Activate your Conda environment (e.g., `conda activate drl-clean`).
#
# 5. Navigate to the directory containing this script and the other required files.
#    Example: `cd C:\path\to\your_project_folder`
#
# 6. Run the script from the terminal, providing the path to your trained model:
#
#    If your agent was trained WITHOUT manual leg length concatenation to observations:
#    python evaluate_sac_morph.py --model_path ./path_to_your_logs/seed_X_run_suffix/models/your_model.ckpt
#
#    If your agent was trained WITH manual leg length concatenation (using the --agent_uses_augmented_obs flag during training):
#    python evaluate_sac_morph.py --model_path ./path_to_your_logs/seed_X_run_suffix/models/your_model.ckpt --agent_uses_augmented_obs
#
#    Replace `./path_to_your_logs/seed_X_run_suffix/models/your_model.ckpt` with the actual path to your
#    trained model checkpoint file.
#
# 7. Optional arguments:
#    --num_eval_runs <N>   : Number of episodes per morphology (default: 10).
#    --seed <S>            : Random seed for evaluation (default: 42).
#    --no_log_video        : Disable video logging.
#    --obs_noise_std <STD> : Standard deviation for Gaussian noise added to observations (default: 0.01).
#
# 8. Output:
#    - Progress will be printed to the terminal.
#    - Videos (if enabled) will be saved in a subdirectory like `evaluation_videos/your_model_name/`.
#    - A `evaluation_summary.txt` file with detailed results will also be saved in that video subdirectory.

import os
import argparse
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import imageio
import time

# Import network definition from your training script
# !!! IMPORTANT: Change 'sac_train_script' if your training script is named differently !!!
from train_obv_morph import Pi_FC
from halfcheetah_morph_env import HalfCheetahMorphEnv

# --- Original Morphology Parameters ---
ORIGINAL_LEG_LENGTHS = {
    'b_thigh_len': 0.145,
    'b_shin_len': 0.150,
    'f_thigh_len': 0.133,
    'f_shin_len': 0.106
}

# --- Evaluation Specific Morphology ---
EVAL_FLOOR_FRICTION = [0.2, 0.1, 0.1]

# --- Helper to augment observation if needed ---
MORPH_PARAM_KEYS = ['b_thigh_len', 'b_shin_len', 'f_thigh_len', 'f_shin_len']

def augment_observation(raw_obs, leg_lengths_dict):
    leg_values = np.array([leg_lengths_dict[key] for key in MORPH_PARAM_KEYS], dtype=np.float64)
    if raw_obs.dtype != np.float64:
        raw_obs = raw_obs.astype(np.float64)
    augmented_obs = np.concatenate((raw_obs, leg_values))
    return augmented_obs

def add_observation_noise(observation, noise_std):
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=observation.shape).astype(observation.dtype)
        return observation + noise
    return observation

def evaluate_agent(arglist):
    print(f"Starting evaluation with args: {arglist}")
    random.seed(arglist.seed)
    np.random.seed(arglist.seed) # Corrected: use arglist.seed
    torch.manual_seed(arglist.seed) # Corrected: use arglist.seed
    if torch.cuda.is_available() and arglist.use_gpu:
        torch.cuda.manual_seed_all(arglist.seed) # Corrected: use arglist.seed
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "half_cheetah.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # --- Determine Observation Size ---
    try:
        with HalfCheetahMorphEnv(xml_path=xml_path, use_custom_morphology=False, render_mode=False) as temp_env:
            base_obs_size = temp_env.observation_space.shape[0]
            action_size = temp_env.action_space.shape[0]
    except Exception as e:
        print(f"Error creating temporary environment for obs_size: {e}")
        raise

    if arglist.agent_uses_augmented_obs:
        actor_obs_size = base_obs_size + 4
        print(f"Agent expects augmented observations. Base obs: {base_obs_size}, Augmented obs: {actor_obs_size}")
    else:
        actor_obs_size = base_obs_size
        print(f"Agent expects raw observations. Obs size: {actor_obs_size}")

    # --- Load Actor Network ---
    actor = Pi_FC(actor_obs_size, action_size).to(device)
    if not os.path.exists(arglist.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {arglist.model_path}")

    try:
        checkpoint = torch.load(arglist.model_path, map_location=device)
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
        elif 'agent_state_dict' in checkpoint:
             actor.load_state_dict(checkpoint['agent_state_dict'])
        else:
            actor.load_state_dict(checkpoint)
        actor.eval()
        print(f"Successfully loaded actor model from: {arglist.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # --- Create Video Directory ---
    model_name_for_dir = os.path.splitext(os.path.basename(arglist.model_path))[0]
    eval_video_dir = os.path.join(script_dir, arglist.video_dir_name, model_name_for_dir)
    os.makedirs(eval_video_dir, exist_ok=True)
    print(f"Saving videos and results to: {eval_video_dir}")

    # --- Define Leg Length Scaling Factors (n) ---
    n_values_segment1 = np.arange(0.1, 1.1 + 1e-5, 0.2)
    n_values_segment2 = np.arange(1.1, 8.1 + 1e-5, 0.5) # Updated as per request
    all_n_values = np.concatenate([n_values_segment1, n_values_segment2])

    max_episode_steps = arglist.max_episode_steps
    results = {}

    for n_scale in all_n_values:
        current_leg_lengths = {key: val * n_scale for key, val in ORIGINAL_LEG_LENGTHS.items()}
        print(f"\n--- Evaluating for n_scale = {n_scale:.2f} ---")
        print(f"Target Leg lengths: {current_leg_lengths}")
        print(f"Floor friction: {EVAL_FLOOR_FRICTION}")
        if arglist.obs_noise_std > 0:
            print(f"Observation noise std: {arglist.obs_noise_std}")

        episode_rewards = []
        env = None

        for run_idx in range(arglist.num_eval_runs):
            actual_leg_lengths_in_env = {} # To store what the env actually used if it clips
            try:
                env = HalfCheetahMorphEnv(
                    xml_path=xml_path,
                    leg_lengths=current_leg_lengths, # Target leg lengths
                    floor_friction=EVAL_FLOOR_FRICTION,
                    use_custom_morphology=True,
                    render_mode=(run_idx == 0 and arglist.log_video)
                )
                # Optional: If your env has a way to report the actual applied leg lengths (e.g., after clipping)
                # you might want to log them. For now, we assume `current_leg_lengths` are applied.
                # if hasattr(env, 'get_current_morphology_parameters'):
                #    actual_leg_lengths_in_env = env.get_current_morphology_parameters()['leg_lengths']
                #    if run_idx == 0: print(f"  Actual leg lengths in env: {actual_leg_lengths_in_env}")


                raw_obs = env.reset()
                # Apply noise *after* potential augmentation
                if arglist.agent_uses_augmented_obs:
                    obs_for_agent = augment_observation(raw_obs, current_leg_lengths)
                else:
                    obs_for_agent = raw_obs.astype(np.float64)
                
                obs_for_agent = add_observation_noise(obs_for_agent, arglist.obs_noise_std)


                current_episode_reward = 0
                frames = []

                for step in range(max_episode_steps):
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs_for_agent, dtype=torch.float64, device=device).unsqueeze(0)
                        action_tensor, _ = actor(obs_tensor, deterministic=True)
                    action = action_tensor.cpu().numpy().squeeze()

                    raw_next_obs, reward, done, info = env.step(action.astype(np.float32))

                    if arglist.agent_uses_augmented_obs:
                        next_obs_for_agent = augment_observation(raw_next_obs, current_leg_lengths)
                    else:
                        next_obs_for_agent = raw_next_obs.astype(np.float64)
                    
                    next_obs_for_agent = add_observation_noise(next_obs_for_agent, arglist.obs_noise_std)

                    obs_for_agent = next_obs_for_agent
                    current_episode_reward += reward

                    if run_idx == 0 and arglist.log_video:
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            frames.append(frame)
                    if done:
                        break
                
                episode_rewards.append(current_episode_reward)
                print(f"  Run {run_idx + 1}/{arglist.num_eval_runs}: Reward = {current_episode_reward:.2f}")

                if run_idx == 0 and arglist.log_video and frames:
                    video_filename = f"eval_n_scale_{n_scale:.2f}_noise_{arglist.obs_noise_std:.3f}.gif"
                    video_path = os.path.join(eval_video_dir, video_filename)
                    imageio.mimsave(video_path, frames, fps=arglist.video_fps)
                    print(f"  Saved video: {video_path}")

            except Exception as e:
                print(f"  Error during evaluation run {run_idx + 1} for n_scale {n_scale:.2f}: {e}")
                import traceback
                traceback.print_exc()
                episode_rewards.append(np.nan)
            finally:
                if env is not None:
                    env.close()
                    env = None

        avg_reward = np.nanmean(episode_rewards) if episode_rewards else np.nan
        std_reward = np.nanstd(episode_rewards) if episode_rewards else np.nan
        successful_runs = len(episode_rewards) - np.sum(np.isnan(episode_rewards))
        results[n_scale] = {'avg_reward': avg_reward, 'std_reward': std_reward, 'all_rewards': episode_rewards, 'successful_runs': successful_runs}
        print(f"--- n_scale = {n_scale:.2f} | Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f} (over {successful_runs} successful runs) ---")

    print("\n\n--- Final Evaluation Summary ---")
    for n_scale, data in results.items():
        print(f"n_scale: {n_scale:.2f} | Avg Reward: {data['avg_reward']:.2f} +/- {data['std_reward']:.2f} | Successful Runs: {data['successful_runs']}/{arglist.num_eval_runs}")

    results_filename = os.path.join(eval_video_dir, f"evaluation_summary_noise_{arglist.obs_noise_std:.3f}.txt")
    with open(results_filename, "w") as f:
        f.write("n_scale,avg_reward,std_reward,successful_runs,all_rewards\n")
        for n_scale, data in results.items():
            rewards_str = ";".join(map(lambda x: f"{x:.2f}" if not np.isnan(x) else "NaN", data['all_rewards']))
            f.write(f"{n_scale:.2f},{data['avg_reward']:.2f},{data['std_reward']:.2f},{data['successful_runs']},\"{rewards_str}\"\n")
    print(f"Saved detailed results to: {results_filename}")


def parse_args():
    parser = argparse.ArgumentParser("SAC Morphology Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained agent .ckpt model file (actor state_dict).")
    parser.add_argument("--agent_uses_augmented_obs", action="store_true", default=False,
                        help="Set this flag if the loaded agent was trained with leg lengths concatenated to observations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation.")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU if available (default True).") # Kept default True
    parser.add_argument("--num_eval_runs", type=int, default=10, help="Number of evaluation episodes per morphology variation.")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max steps per evaluation episode.")
    
    # Video logging arguments
    parser.add_argument("--log_video", action=argparse.BooleanOptionalAction, default=True, help="Log videos of the first run. Use --no-log-video to disable.")
    parser.add_argument("--video_dir_name", type=str, default="evaluation_videos", help="Directory name to save videos, relative to script location.")
    parser.add_argument("--video_fps", type=int, default=30, help="FPS for logged videos.")
    
    # Observation noise argument
    parser.add_argument("--obs_noise_std", type=float, default=0.01,
                        help="Standard deviation of Gaussian noise to add to observations. Set to 0 for no noise (default: 0.01).")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate_agent(args)