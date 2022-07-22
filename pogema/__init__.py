from gym import register
from pogema.grid_config import GridConfig
from pogema.integrations.make_pogema import pogema_v0

from pogema.grid_config import Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8
from pogema.grid_config import Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16
from pogema.grid_config import Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32
from pogema.grid_config import Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64

__version__ = '1.1.0'

__all__ = [
    'GridConfig',
    'pogema_v0',
    'Easy8x8', 'Normal8x8', 'Hard8x8', 'ExtraHard8x8',
    'Easy16x16', 'Normal16x16', 'Hard16x16', 'ExtraHard16x16',
    'Easy32x32', 'Normal32x32', 'Hard32x32', 'ExtraHard32x32',
    'Easy64x64', 'Normal64x64', 'Hard64x64', 'ExtraHard64x64',
]

register(
    id="Pogema-v0",
    entry_point="pogema.integrations.make_pogema:make_single_agent_gym",
)

# def main():
#     q = _get_num_agents_by_target_density(64, 0.0223, 0.3)
#     print(_get_target_density_by_num_agents(64, q, 0.3))
#     print(q)


# if __name__ == '__main__':
#     for env, Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64
#     main()

# for size, max_episode_steps in zip([8, 16, 32, 64], [64, 128, 256, 512]):
#     for obstacle_density in [0.3]:
#         for difficulty, agent_density in zip(['easy', 'normal', 'hard', 'extra-hard'],
#                                              [0.0223, 0.0446, 0.0892, 0.1784]):
#             num_agents = _get_num_agents_by_target_density(size, agent_density, obstacle_density)
#             register(
#                 id=f'Pogema-{size}x{size}-{difficulty}-v0',
#                 entry_point="pogema.integrations.make_pogema:make_pogema",
#                 kwargs={"grid_config": GridConfig(size=size, num_agents=num_agents, density=obstacle_density,
#                                                   max_episode_steps=max_episode_steps),
#                         "integration": None})
