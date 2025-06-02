# DRL Final Project: Robust HalfCheetah Locomotion under Variation

## Overview
This project explores the robustness and generalization capabilities of Soft Actor-Critic (SAC) agents trained in the MuJoCo HalfCheetah environment. Agents are tested against domain shifts in morphology (leg length) and environment (ground friction, observation noise).

## Directory Structure and File Descriptions

### Environment Configuration
- **`environment.yml`**: Defines the full conda environment for the project, including Python version and key packages (e.g., torch, gymnasium, mujoco, wandb).
- **`requirements.txt`**: A pip-installable list of Python dependencies for broader compatibility.

### MuJoCo Model
- **`half_cheetah.xml`**: The base XML file defining the HalfCheetah morphology, joint structure, and actuators. Used as the template for all simulations.

### Environment Variants
- **`halfcheetah_custom_env.py`**: Contains functions to modify the MuJoCo XML on the fly, adjusting leg segment lengths and ground friction dynamically.
- **`halfcheetah_morph_env.py`**: Defines a custom Gymnasium environment where the HalfCheetah morphology is randomized on each episode reset.

### Training Scripts
- **`train_vanilla_noise_morph.py`**: Trains SAC with observation noise and per-reset random leg morphologies.
- **`train_obv_morph.py`**: Trains SAC with only morphology changes (no noise).
- **`train_obv_noise_morph.py`**: Trains SAC with both observation noise and morphology perturbations.
- **`halfcheetah_vanilla_morph.py`**: Trains SAC with moderate random morphology and friction.
- **`halfcheetah_vanilla_morph_exac.py`**: Trains SAC with random morphology and friction.
- **`halfcheetah_train_curriculur.py`**: Trains SAC with curriculum-based gradual randomization of morphology and friction. The curriculum progresses in four stages, beginning with default morphology and incrementally increasing variation to improve robustness.

These scripts allow control via command-line flags:
```bash
python train_obv_morph.py --wandb --video-freq 50
```
- `--wandb`: Enable Weights & Biases logging
- `--video-freq`: Save evaluation video every N episodes

### Evaluation Script
- **`evaluate_sac_morph.py`**: Evaluates a trained SAC policy under varying test conditions (e.g., reduced/increased leg length, noisy sensors) and plots reward curves. Use this script to assess robustness.

```bash
python evaluate_sac_morph.py --model-path ./checkpoints/best_model.pth
```

## How to Use

### 1. Environment Setup
**Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate drl-clean
```

**Or pip:**
```bash
pip install -r requirements.txt
```

Make sure MuJoCo is properly installed and licensed.

### 2. Training
Choose a script depending on your setup:
- Run `train_obv_morph.py` to train on morphology variation only.
- Run `train_vanilla_noise_morph.py` to include observation noise.

Each script logs training metrics and optionally videos.

### 3. Evaluation
After training, use `evaluate_sac_morph.py` to test the model under diverse conditions and generate performance plots.

### 4. Model Variations
Modify leg length ranges and friction coefficients in the environment code or XML as needed.

## Dependencies
- Python 3.10
- PyTorch 2.0.1+cu117
- MuJoCo 3.3.2
- Gymnasium, WandB, Matplotlib, MoviePy, Seaborn

## Notes
- Temporary MuJoCo XMLs are deleted after being loaded.
- Training and evaluation scripts automatically use the correct environment wrappers.

## License
This codebase is for academic and educational use only. MuJoCo license terms apply.