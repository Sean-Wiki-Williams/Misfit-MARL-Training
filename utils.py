import torch

# Export a deterministic TorchScript actor. torch.jit.script cannot compile TorchRL's
# ProbabilisticActor, so rebuild the shared-parameter MLP as plain Linear/Tanh layers.
class DeterministicActor(torch.nn.Module):
    def __init__(self, mlp, low, high):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("low", low.clone())
        self.register_buffer("high", high.clone())

    def forward(self, obs):
        loc, _ = self.mlp(obs).chunk(2, dim=-1)
        return self.low + (torch.tanh(loc) + 1) / 2 * (self.high - self.low)


def build_deterministic_actor(policy_obj):
    state = policy_obj._policy_net[0].state_dict()
    layers = []
    for i in (0, 2, 4):
        weight, bias = state[f"params.{i}.weight"], state[f"params.{i}.bias"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(weight.device)
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if i < 4:
            layers.append(torch.nn.Tanh())
    device = layers[0].weight.device
    return DeterministicActor(
        torch.nn.Sequential(*layers), policy_obj.low.to(device), policy_obj.high.to(device)
    ).eval()

def load_actor(path="models/actor/mappo_actor_scripted.pt"):
    actor = torch.jit.load(path)
    actor.eval()
    return actor