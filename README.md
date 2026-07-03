# Multi-Agent Reinforcement Learning for Knights, Archers, Zombies

This project trains multiple agents in PettingZoo's Knights, Archers, Zombies environment using independent learners per agent.

Implemented learners:
- DQN (value-based)
- REINFORCE (policy gradient)
- A2C (actor-critic)
- Random baseline

## Environment

- Environment: `pettingzoo.butterfly.knights_archers_zombies_v10`
- API mode: parallel environment
- Default environment settings in `main.py`:
  - `num_archers=2`
  - `num_knights=2`
  - `spawn_rate=1`
  - `max_zombies=1`
  - `max_arrows=2`
  - `max_cycles=2000`

## Repository Structure

- `main.py`: training entrypoint, CLI prompts, TensorBoard launcher, training loops
- `DQN.py`: `DQN_Logic` implementation
- `REINFORCE.py`: `ReinforcePolicy` implementation
- `A2C.py`: `A2C_Logic` implementation
- `PPO.py`: PPO policy class (currently not wired into `main.py` CLI)
- `requirements.txt`: project dependencies

Generated during runs:
- `runs/`: TensorBoard event files
- `logs/`: verbose run logs (when enabled)
- `checkpoints/`: saved model checkpoints (`*.pt`)

## Setup

### 1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

Notes:
- On Windows, `requirements.txt` pulls the CUDA 12.1 PyTorch wheel.
- If you do not have a CUDA-compatible GPU/driver, install CPU PyTorch manually first, then install the rest.

## Run Training

```powershell
python main.py
```

You will be prompted for:
- Algorithm: `dqn`, `pg`, `a2c`, or `random`
- Whether to load saved model weights from `checkpoints/`
- Number of episodes
- Headless mode
- Verbose logging
- Initial epsilon (DQN only)

Example prompt flow:

```text
Select algorithm [dqn/pg/a2c/random] (default dqn): dqn
Load saved model on start? [y/N]: n
Enter number of episodes (default 3000): 1000
Run headless? [Y/n]: y
Enable verbose logging? [y/N]: n
Set initial epsilon for DQN (default 1.0): 1.0
```

## Checkpoints

Checkpoints are automatically saved at the end of training (including KeyboardInterrupt handling):
- DQN: `checkpoints/dqn_<agent_name>.pt`
- REINFORCE: `checkpoints/reinforce_<agent_name>.pt`
- A2C: `checkpoints/a2c_<agent_name>.pt`

If "Load saved model on start?" is enabled, matching checkpoint files are loaded per agent if found.

## TensorBoard

`main.py` starts TensorBoard automatically on port `6006` and attempts to open a browser tab.

Log directories are created per run, for example:
- `runs/DQN_MM_DD_HHMM`
- `runs/REINFORCE_MM_DD_HHMM`
- `runs/A2C_MM_DD_HHMM`
- `runs/RANDOM_MM_DD_HHMM`

If auto-open fails, open manually:

```text
http://localhost:6006
```

## Current CLI Algorithms

- `dqn`: trains independent DQN policies with replay memory and target networks
- `pg`: trains independent REINFORCE policies
- `a2c`: trains independent actor-critic policies with rollout updates
- `random`: random-action baseline for comparison

## Notes and Limitations

- `PPO.py` exists, but PPO is not currently selectable from the `main.py` CLI.
- Training/log/checkpoint folders are ignored in git by default (`.gitignore`).

## Troubleshooting

- TensorBoard command not found:
  - Install it in your active venv: `pip install tensorboard`
- No render window appears:
  - Disable headless mode when prompted
- Slow training:
  - Start with fewer episodes and headless mode enabled
- No checkpoints loaded:
  - Verify algorithm-specific checkpoint name prefix (`dqn_`, `reinforce_`, `a2c_`)

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
