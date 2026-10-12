from torchrl.modules import MultiAgentMLP
from tensordict.nn import TensorDictModule
import torch

class mappo_critic:
    def __init__(self, env):
        self._env_sample = env

        _critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
            centralised=True, #True for MAPPO
            share_params=True,
            device=self._env_sample.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        _critic = TensorDictModule(
            module=_critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )

        self.policy = _critic