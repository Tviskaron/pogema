import numpy as np
import gym
from gym.error import ResetNeeded
from gym.spaces import Box

from pogema.grid import Grid
from pogema.grid_config import GridConfig
from pogema.wrappers.matrix_observation import MatrixObservationWrapper, MatrixOnlyObservationWrapper
from pogema.wrappers.metrics import MetricsWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit


class ActionsSampler:
    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class FullStateNotAvailableError(Exception):
    pass


class PogemaBase(gym.Env):

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.config = grid_config

        full_size = self.config.obs_radius * 2 + 1
        self.action_space = gym.spaces.Discrete(len(self.config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.config.seed)

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    def _get_agents_obs(self, agent_id=0):
        return np.concatenate([
            self.grid.get_obstacles_for_agent(agent_id)[None],
            self.grid.get_positions(agent_id)[None],
            self.grid.get_square_target(agent_id)[None]
        ])

    def check_reset(self):
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        return self._multi_action_sampler.sample_actions(dim=self.config.num_agents)

    def get_num_agents(self):
        return self.config.num_agents

    def _obs(self):

        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.config.num_agents):
            result = {}
            result['obstacles'] = self.grid.get_obstacles_for_agent(agent_idx)

            result['agents'] = self.grid.get_positions(agent_idx)
            result['xy'] = agents_xy_relative[agent_idx]
            result['target_xy'] = targets_xy_relative[agent_idx]
            results.append(result)
        return results

    def get_obstacles(self, ignore_borders=False):
        raise FullStateNotAvailableError

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return FullStateNotAvailableError

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return FullStateNotAvailableError

    def get_state(self, ignore_borders=False, as_dict=False):
        return FullStateNotAvailableError


# class PogemaCoopFinish(PogemaBase):
#     def __init__(self, config=GridConfig(num_agents=2)):
#         super().__init__(config)
#         self.num_agents = self.config.num_agents
#         self.is_multiagent = True
#         self.active = None
#
#     def _obs(self):
#         return [self._get_agents_obs(index) for index in range(self.config.num_agents)]
#
#     def step(self, action: list):
#         assert len(action) == self.config.num_agents
#         rewards = []
#
#         infos = [dict() for _ in range(self.config.num_agents)]
#
#         dones = []
#         for agent_idx in range(self.config.num_agents):
#             agent_done = self.grid.move(agent_idx, action[agent_idx])
#
#             if agent_done:
#                 rewards.append(1.0)
#             else:
#                 rewards.append(0.0)
#
#             dones.append(agent_done)
#
#         obs = self._obs()
#         return obs, rewards, dones, infos
#
#     def reset(self):
#         self.grid: CooperativeGrid = CooperativeGrid(grid_config=self.config)
#         self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
#         return self._obs()


class Pogema(PogemaBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.active = None

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = []

        used_cells = {}

        agents_xy = self.grid.get_agents_xy()
        blocked = 'blocked'
        visited = 'visited'

        for agent_idx, (x, y) in enumerate(agents_xy):
            if self.active[agent_idx]:
                dx, dy = self.config.MOVES[action[agent_idx]]
                used_cells[x + dx, y + dy] = blocked if (x + dx, y + dy) in used_cells else visited
                used_cells[x, y] = blocked
        for agent_idx in range(self.config.num_agents):
            if self.active[agent_idx]:
                x, y = agents_xy[agent_idx]
                dx, dy = self.config.MOVES[action[agent_idx]]
                if used_cells.get((x + dx, y + dy), None) != blocked:
                    self.grid.move(agent_idx, action[agent_idx])

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            dones.append(on_goal)

        for agent_idx in range(self.config.num_agents):
            infos[agent_idx]['is_active'] = self.active[agent_idx]

            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.active[agent_idx] = False

        obs = self._obs()
        return obs, rewards, dones, infos

    def reset(self, **kwargs):
        self.grid: Grid = Grid(grid_config=self.config)
        self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
        return self._obs()


def _make_pogema(grid_config):
    env = Pogema(grid_config=grid_config)
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    env = MetricsWrapper(env)

    return env
