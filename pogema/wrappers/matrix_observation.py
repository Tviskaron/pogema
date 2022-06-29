import gym
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from numpy import float32


class MatrixObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        full_size = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    @staticmethod
    def get_square_target(x, y, tx, ty, obs_radius):
        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        dx, dy = x - tx, y - ty

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result

    @staticmethod
    def to_matrix(observations):
        result = []
        obs_radius = observations[0]['obstacles'].shape[0] // 2
        for agent_idx, obs in enumerate(observations):
            result.append(
                {"obs": np.concatenate([obs['obstacles'][None], obs['agents'][None],
                                        MatrixObservationWrapper.get_square_target(*obs['xy'], *obs['target_xy'],
                                                                                   obs_radius)[None]]).astype(float32),
                 "xy": np.array(obs['xy'], dtype=float32),
                 "target_xy": np.array(obs['target_xy'], dtype=float32),
                 })
        return result

    def observation(self, observation):
        result = self.to_matrix(observation)
        return result


class MatrixOnlyObservationWrapper(MatrixObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        full_size = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size))

    def observation(self, observation):
        return [obs['obs'] for obs in super().observation(observation)]
