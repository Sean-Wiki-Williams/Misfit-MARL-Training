
import torch
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

class mappo_policy:
    def __init__(self, env):
        self._env_sample = env

        self._policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # n_obs_per_agent
                n_agent_outputs=2 * self._env_sample.full_action_spec[self._env_sample.action_key].shape[-1],  # 2 * mean & scale for each action
                n_agents=self._env_sample.n_agents, 
                centralised=False,  # Each agent will act from its observation
                share_params=True, # All agents will use the same policy
                device=self._env_sample.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )

        # Wraps the policy network into a TensorDictModule to put the policy into a format compatible with TorchRL's TensorDict-based API
        self._policy_module = TensorDictModule(
            self._policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")], #loc is the mean of the action distribution
            #scale is the standard deviation of the action distribution
        )

        self.action_spec = self._env_sample.full_action_spec_unbatched[self._env_sample.action_key]

        self.low = self.action_spec.space.low    # shape [action_dim]
        self.high = self.action_spec.space.high  # shape [action_dim]

        self.policy = ProbabilisticActor(
            module=self._policy_module,
            spec=self._env_sample.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[self._env_sample.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": self.low, "high": self.high},
            return_log_prob=True,
        )
        

