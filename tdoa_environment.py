import typing
from typing import Dict, List

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    Color,
    LINEAR_FRICTION,
    ScenarioUtils,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class TdoaScenario(BaseScenario):
    """
    Multi-agent cooperative localization using Time Difference of Arrival (TDoA).
    Agents move to reduce the uncertainty ellipse of a target emitting a signal.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        # Number of agents
        self.n_agents = kwargs.pop("n_agents", 4)

        # World size (km)
        self.world_spawning_x = kwargs.pop("world_spawning_x", 40.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 40.0)

        # Signal model
        self.signal_speed = kwargs.pop("signal_speed", 0.3)  # km/s (speed of sound ~0.343 km/s)
        self.tdoa_noise = kwargs.pop("tdoa_noise", 0.01)     # seconds of noise

        # Reward shaping
        self.shared_rew = True
        self.prev_area = None

        # Agent dynamics
        self.dt = kwargs.pop("dt", 5.0)
        self.max_speed = kwargs.pop("max_speed", 0.15)
        self.min_speed = kwargs.pop("min_speed", 0.08)
        self.agent_radius = kwargs.pop("agent_radius", 1.0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # World
        world = World(
            batch_dim,
            device,
            substeps=5,
            collision_force=500.0,
            dt=self.dt,
            gravity=(0.0, 0.0),
            drag=0.0,
            linear_friction=LINEAR_FRICTION,
            angular_friction=ANGULAR_FRICTION,
        )

        # Agents
        known_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]

        for i in range(self.n_agents):
            color = known_colors[i % len(known_colors)]
            max_steering_angle = torch.pi / 4
            width = self.agent_radius

            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=color,
                render_action=True,
                shape=Box(length=self.agent_radius * 2, width=width),
                u_range=[self.max_speed, max_steering_angle],
                u_multiplier=[1.0, 1.0],
                dynamics=KinematicBicycle(
                    world,
                    width=width,
                    l_f=self.agent_radius,
                    l_r=self.agent_radius,
                    max_steering_angle=max_steering_angle,
                ),
            )

            world.add_agent(agent)

        # Target landmark
        self.target = Landmark(
            name="target",
            collide=False,
            color=Color.RED,
            shape=Sphere(radius=1.5),
        )
        world.add_landmark(self.target)

        return world

    def reset_world_at(self, env_index: int = None):
        # Spawn agents and target randomly
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + [self.target],
            self.world,
            env_index,
            min_distance=5.0,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        # Reset ellipse area memory
        if env_index is None:
            self.prev_area = torch.zeros(self.world.batch_dim, device=self.world.device)
        else:
            self.prev_area[env_index] = 0.0

    # -------------------------------------------------------------
    # TDoA Geometry
    # -------------------------------------------------------------
    def compute_tdoa(self):
        """
        Compute arrival times and TDoA matrix.
        """
        agent_positions = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        target_pos = self.target.state.pos.unsqueeze(1)

        distances = torch.linalg.vector_norm(agent_positions - target_pos, dim=-1)
        arrival_times = distances / self.signal_speed

        # Add noise
        arrival_times += torch.randn_like(arrival_times) * self.tdoa_noise

        return arrival_times

    def compute_ellipse_area(self, arrival_times):
        """
        Approximate ellipse area from TDoA geometry.
        We use covariance of arrival times as a proxy for uncertainty.
        """
        # Covariance across agents
        cov = torch.var(arrival_times, dim=1)  # shape [batch]
        # Convert to ellipse area proxy
        area = torch.pi * cov  # simple proxy
        return area

    # -------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------
    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            arrival_times = self.compute_tdoa()
            area = self.compute_ellipse_area(arrival_times)

            # Reward = shrink ellipse
            if self.prev_area is None:
                self.prev_area = area.clone()

            self.rew = self.prev_area - area
            self.prev_area = area.clone()

        return self.rew

    # -------------------------------------------------------------
    # Action processing
    # -------------------------------------------------------------
    def process_action(self, agent: Agent):
        u = agent.action.u.clone()
        fraction = (u[:, 0] + self.max_speed) / (2 * self.max_speed)
        u[:, 0] = self.min_speed + fraction.clamp(0.0, 1.0) * (self.max_speed - self.min_speed)
        agent.action.u = u

    # -------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------
    def observation(self, agent: Agent):
        wx, wy = self.world_spawning_x, self.world_spawning_y
        scale = torch.tensor([wx, wy], device=agent.state.pos.device)

        # Compute arrival time for this agent
        distance = torch.linalg.vector_norm(agent.state.pos - self.target.state.pos, dim=-1)
        arrival_time = distance / self.signal_speed
        arrival_time += torch.randn_like(arrival_time) * self.tdoa_noise

        parts = [
            agent.state.pos / scale,
            agent.state.vel / self.max_speed,
            arrival_time.unsqueeze(-1),
        ]
        return torch.cat(parts, dim=-1)

    # -------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------
    def done(self) -> Tensor:
        # Never terminates early
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

    # -------------------------------------------------------------
    # Info
    # -------------------------------------------------------------
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "ellipse_reward": self.rew,
        }

    # -------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List["Geom"] = []

        # Draw target
        geoms.append(ScenarioUtils.plot_entity_rotation(self.target, env_index))

        return geoms
