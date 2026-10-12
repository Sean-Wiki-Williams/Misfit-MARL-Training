# Misfit MARL Training

Multi-agent reinforcement learning with **MAPPO** (multi-agent PPO) on a custom [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) scenario. Training is built on [TorchRL](https://github.com/pytorch/rl) and runs many environments in parallel on the GPU.

## Scenario

Defined in `environment.py` (`TrainingScenario`):

- Fixed-wing-style aircraft (kinematic bicycle dynamics), no sensors, each with its own goal; positions are random each episode
- Units: 1 world unit = 1 km, 1 s of sim time. The world is 80 km x 80 km (~50 mi across)
- Each step is 5 s. Speed is limited to 0.08–0.15 km/s (~290–540 km/h): aircraft cannot stop or reverse. At full speed an aircraft covers ~0.75 km per step and ~75 km in a 100-step episode. Minimum turn radius is 2 km
- Aircraft icons (0.5 km radius) and goals (1.5 km radius) are exaggerated so they are visible in renders; the goal radius is also the "reached" tolerance
- Observation per agent (9 values, normalised to roughly -1..1): vector to its own goal (2), position (2), velocity (2), heading as cos and sin (2), angular velocity (1)
- Action per agent: continuous (speed command, steering). The speed command is mapped onto the 0.08–0.15 km/s flight envelope
- Reward per agent: progress toward its goal + a bonus of `5.0` when all agents reach their goals + a `-0.1` penalty for collisions with other agents
- Episode ends when every agent is on its goal, or after 100 steps

## Algorithm

- `mappo_policy.py`: decentralised actor. A shared-parameter MLP (2 x 256, Tanh) maps each agent's own observation to a `TanhNormal` action distribution.
- `mappo_critic.py`: centralised critic. A shared-parameter MLP (2 x 256, Tanh) that sees all agents' observations and outputs one value per agent.
- `main.py`: training loop using `ClipPPOLoss` with GAE (`gamma=0.99`, `lmbda=0.9`, `clip_epsilon=0.2`), Adam (`lr=5e-5`) and gradient clipping.

Key hyperparameters are at the top of `main.py`:

| Setting | Default |
|---|---|
| Parallel environments | 600 |
| `frames_per_batch` | 6,000 |
| `n_iters` | 100 |
| `num_epochs` per iteration | 10 |
| `minibatch_size` | 2,000 |

## Repository Structure

- `main.py`: training entrypoint
- `environment.py`: VMAS scenario
- `mappo_policy.py`: actor network
- `mappo_critic.py`: critic network
- `requirements.txt`: dependencies

Generated during runs (git-ignored):

- `runs/`: TensorBoard logs
- `checkpoints/`: saved models (`checkpoint_<step>.pt`)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 build of PyTorch. Verify the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False` and the version contains `+cpu`, a CPU-only PyTorch is installed. Reinstall with:

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The script falls back to the CPU if CUDA is unavailable, but 600 parallel environments will be slow.

## Run Training

```powershell
python main.py
```

## Checkpoints

Every 100 iterations (`checkpoint_interval`), `main.py` saves `checkpoints/checkpoint_<step>.pt` containing the actor and critic weights, optimizer state and step number.

## TensorBoard

Logs go to `runs/mappo_<timestamp>`. The program does not launch TensorBoard; start it in a second terminal:

```powershell
python -m tensorboard.main --logdir runs
```

Then open http://localhost:6006. Logged data:

- **Scalars**: losses (total, objective, critic, entropy), rewards (step mean, running return, episode return, reward components), completed episodes, mean action scale, frames processed
- **Histograms**: actor and critic weights, every `histogram_interval` (5) iterations
- **Images**: after training, a rollout from a fresh random environment is rendered and logged as a looping GIF (`render/final_rollout`). The GIF is also saved as `final_rollout.gif` in the run's log folder.

## Troubleshooting

- `AssertionError: Torch not compiled with CUDA enabled`: CPU-only PyTorch is installed; see Setup.
- The progress bar sits at the last iteration for a while: the final rollout is being rendered.
- `add_video` fails on Windows (`moviepy` temp-file permission error): that is why the GIF is written to the event file directly.
