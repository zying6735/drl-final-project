# halfcheetah_morph_env.py

import os
import gym
import numpy as np
from gym import spaces
import mujoco
from mujoco import MjModel, MjData
from halfcheetah_custom_env import modify_all_leg_lengths_and_friction

class HalfCheetahMorphEnv(gym.Env):
    def __init__(self, xml_path, leg_lengths=None, floor_friction=None, use_custom_morphology=True, render_mode=False):
        super().__init__()

        with open(xml_path, 'r') as f:
            xml_content = f.read()

        if use_custom_morphology and leg_lengths is not None and floor_friction is not None:
            modified_xml = modify_all_leg_lengths_and_friction(
                xml_content,
                **leg_lengths,
                floor_friction=floor_friction
            )
            if modified_xml is None:
                raise RuntimeError("Failed to modify XML for morphology randomization.")
            xml_to_load = modified_xml
        else:
            # Use default morphology (no XML modifications)
            xml_to_load = xml_content

        self.model = MjModel.from_xml_string(xml_to_load)
        self.data = MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if self.render_mode:
            self.renderer = mujoco.Renderer(self.model, width=500, height=480)
            self.frames = []
        else:
            self.renderer = None
            self.frames = None

        self.ctrl_cost_weight = 0.1


    def reset(self):
        # Randomize initial state slightly
        qpos = self.model.key_qpos.copy() if self.model.nkey > 0 else np.zeros(self.model.nq)
        qvel = self.model.key_qvel.copy() if self.model.nkey > 0 else np.zeros(self.model.nv)
        qpos += np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel += np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        x_position_before = self.data.qpos[0]

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        #mujoco.mj_step(self.model, self.data)
        for _ in range(5):  # emulate frame_skip=5
            mujoco.mj_step(self.model, self.data)

        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / (self.model.opt.timestep * 5)

        # Forward reward
        reward_run = x_velocity
        # Control cost
        reward_ctrl = -self.ctrl_cost_weight * np.square(action).sum()
        # Total reward
        reward = reward_run + reward_ctrl
        done = False

        # Build the info dictionary
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
        }

        obs = self._get_obs()
        return obs, reward, done, info


    def _compute_reward_and_done(self, action):
        # Forward reward (x-direction velocity of the torso)
        torso_x_vel = self.data.qvel[0]
        reward_run = torso_x_vel

        # Control cost
        reward_ctrl = -self.ctrl_cost_weight * np.square(action).sum()

        # Total reward
        reward = reward_run + reward_ctrl

        # Never-ending episode by default
        done = False
        return reward, done

    def render(self, mode='human'):
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="track")
            img = self.renderer.render()
            if self.frames is not None:
                self.frames.append(img.copy())  # Save frame for .gif!
            if mode == 'rgb_array':
                return img
        else:
            raise RuntimeError("Renderer is not initialized (set render_mode=True to enable it).")


    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

'''environment test script
# Example usage (will not run when imported as a module)
if __name__ == "__main__":
    import imageio

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "half_cheetah.xml")

    leg_lengths = dict(
        b_thigh_len=0.8,
        b_shin_len=0.4,
        f_thigh_len=0.6,
        f_shin_len=0.3
    )
    floor_friction = [0.5, 0.5, 0.5]

    env = HalfCheetahMorphEnv(
        xml_path=xml_path,
        leg_lengths=leg_lengths,
        floor_friction=floor_friction,
        use_custom_morphology=True,  # True to enable custom morphology
        render_mode=True
    )

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Custom env - Mujoco timestep:", env.model.opt.timestep)
    print("Custom env - Effective dt per step (frame_skip=5 emulated):", env.model.opt.timestep * 5)


    obs = env.reset()
    rewards = []
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        # Print reward & info every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: Reward = {reward:.4f}, Info: {info}")

        env.render()

    # Save the .gif
    if env.frames is not None and len(env.frames) > 0:
        gif_path = os.path.join(script_dir, "halfcheetah_test_run.gif")
        imageio.mimsave(gif_path, env.frames, fps=30)
        print(f"Saved 1000-step rollout as GIF: {gif_path}")

    env.close()
'''
