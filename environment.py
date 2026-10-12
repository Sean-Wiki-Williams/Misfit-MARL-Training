import typing
from typing import Dict, List

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    Color,
    DRAG,
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
        self.n_obstacles = kwargs.pop("n_obstacles", 2)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)

        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0.0)
        self.lidar_range = kwargs.pop("lidar_range", 0.3)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_rew = kwargs.pop("shared_rew", False)
        self.final_reward = kwargs.pop("final_reward", 5.0)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)

        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # World
        world = World(
            batch_dim,
            device,
            substeps=5,
            collision_force=500.0,
            dt=0.1,
            gravity=(0.0, 0.0),
            drag=DRAG,
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

            sensors = [
                Lidar(
                    world,
                    n_rays=self.n_lidar_rays,
                    max_range=self.lidar_range,
                    entity_filter=lambda e: isinstance(e, Agent),
                    angle_start=0.0,
                    angle_end=2 * torch.pi,
                )
            ]

            max_steering_angle = torch.pi / 4
            width = self.agent_radius

            agent = Agent(
                name=f"car_{i}",
                collide=True,
                color=color,
                render_action=True,
                sensors=sensors,
                shape=Box(length=self.agent_radius * 2, width=width),
                u_range=[1.0, max_steering_angle],
                u_multiplier=[0.5, 1.0],
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
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_landmark(goal)
            agent.goal = goal
            self.goals.append(goal)

        # Obstacles
        self.obstacles: List[Landmark] = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                color=Color.BLACK,
                shape=Sphere(radius=self.agent_radius * 2 / 3),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.all_goal_reached = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.obstacles + self.goals,
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
                a.on_goal = distance_to_goal < a.shape.circumscribed_radius()

                a.pos_rew = a.goal_dist - distance_to_goal
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
                for b in self.obstacles:
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        mask = distance <= self.min_collision_distance
                        a.agent_collision_rew[mask] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def observation(self, agent: Agent):
        parts = [
            agent.state.pos - agent.goal.state.pos,
            *[agent.state.pos - o.state.pos for o in self.obstacles],
            *[s._max_range - s.measure() for s in agent.sensors],
            agent.state.pos,
            agent.state.vel,
            agent.state.rot,
            agent.state.ang_vel,
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
