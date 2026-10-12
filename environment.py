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


class TrainingScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.plot_grid = False

        # Cars only
        self.n_agents = kwargs.pop("n_agents", 4)

        # Units: 1 world unit = 1 km, 1 time unit = 1 s. The default world is 80 km x 80 km (~50 mi).
        self.world_spawning_x = kwargs.pop("world_spawning_x", 40.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 40.0)

        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0.0)

        self.shared_rew = kwargs.pop("shared_rew", False)
        self.final_reward = kwargs.pop("final_reward", 5.0)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)

        # Aircraft: 0.15 km/s (~540 km/h) max speed, 5 s per step. The icon/keep-out radius is
        # exaggerated so aircraft are visible in renders; it also sets the minimum turn radius
        # (wheelbase / tan(max steering) = 2 km).
        self.dt = kwargs.pop("dt", 5.0)
        self.max_speed = kwargs.pop("max_speed", 0.15)
        self.min_speed = kwargs.pop("min_speed", 0.08)  # stall speed (~290 km/h)
        self.agent_radius = kwargs.pop("agent_radius", 1.0)
        self.goal_radius = kwargs.pop("goal_radius", 2.5)
        self.min_distance_between_entities = self.agent_radius * 2 + self.goal_radius
        self.min_collision_distance = 0.005

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

        # Agents (cars only)
        known_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        self.goals: List[Landmark] = []

        for i in range(self.n_agents):
            color = known_colors[i] if i < len(known_colors) else colors[i - len(known_colors)]

            max_steering_angle = torch.pi / 4
            width = self.agent_radius

            agent = Agent(
                name=f"car_{i}",
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

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()

            world.add_agent(agent)

            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
                shape=Sphere(radius=self.goal_radius),
            )
            world.add_landmark(goal)
            agent.goal = goal
            self.goals.append(goal)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.all_goal_reached = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.goals,
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        for agent in self.world.agents:
            if env_index is None:
                agent.goal_dist = torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos, dim=-1
                )
            else:
                agent.goal_dist[env_index] = torch.linalg.vector_norm(
                    agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0.0
            self.final_rew[:] = 0.0

            for a in self.world.agents:
                a.agent_collision_rew[:] = 0.0
                distance_to_goal = torch.linalg.vector_norm(
                    a.state.pos - a.goal.state.pos, dim=-1
                )
                a.on_goal = distance_to_goal < self.goal_radius

                # Progress is normalised by world size so rewards stay O(1) at any scale
                a.pos_rew = (a.goal_dist - distance_to_goal) / self.world_spawning_x
                a.goal_dist = distance_to_goal
                self.pos_rew += a.pos_rew

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )
            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        mask = distance <= self.min_collision_distance
                        a.agent_collision_rew[mask] += self.agent_collision_penalty
                        b.agent_collision_rew[mask] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def process_action(self, agent: Agent):
        # Fixed wing cannot stop or reverse: map the speed command from [-max, max] to [min, max]
        u = agent.action.u.clone()
        fraction = (u[:, 0] + self.max_speed) / (2 * self.max_speed)
        u[:, 0] = self.min_speed + fraction.clamp(0.0, 1.0) * (self.max_speed - self.min_speed)
        agent.action.u = u

    def observation(self, agent: Agent):
        # Normalised so network inputs are O(1): positions by world size, velocity by max speed,
        # angular velocity by the maximum turn rate.
        wx, wy = self.world_spawning_x, self.world_spawning_y
        scale = torch.tensor([wx, wy], device=agent.state.pos.device)
        max_turn_rate = self.max_speed / (2 * self.agent_radius)
        parts = [
            (agent.goal.state.pos - agent.state.pos) / scale,
            agent.state.pos / scale,
            agent.state.vel / self.max_speed,
            torch.cos(agent.state.rot),
            torch.sin(agent.state.rot),
            agent.state.ang_vel / max_turn_rate,
        ]
        return torch.cat(parts, dim=-1)

    def done(self) -> Tensor:
        return self.all_goal_reached

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collision_rew": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List["Geom"] = []

        # Plot car rotations
        for agent in self.world.agents:
            geoms.append(ScenarioUtils.plot_entity_rotation(agent, env_index))

        # Optional communication lines
        if self.comms_rendering_range > 0:
            for i, agent1 in enumerate(self.world.agents):
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    agent_dist = torch.linalg.vector_norm(
                        agent1.state.pos - agent2.state.pos, dim=-1
                    )
                    if agent_dist[env_index] <= self.comms_rendering_range:
                        color = Color.BLACK.value
                        line = rendering.Line(
                            (agent1.state.pos[env_index]),
                            (agent2.state.pos[env_index]),
                            width=1,
                        )
                        line.set_color(*color)
                        geoms.append(line)

        return geoms
