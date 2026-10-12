import torch

def load_actor(path="models/actor/mappo_actor_scripted.pt"):
    actor = torch.jit.load(path)
    actor.eval()
    return actor