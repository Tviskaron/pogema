import gym


class MetricsWrapper(gym.Wrapper):
    def __init__(self, env, group_name='metrics'):
        super().__init__(env)
        self._ISR = None
        self._group_name = group_name
        self._ep_length = None
        self._steps = None
        self._clear_stats()

    def update_group_name(self, group_name):
        self._group_name = group_name

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        self._steps += 1
        for agent_idx in range(self.env.get_num_agents()):
            infos[agent_idx][self._group_name] = infos[agent_idx].get(self._group_name, {})

            if done[agent_idx]:
                infos[agent_idx][self._group_name].update(Done=True)
                if agent_idx not in self._ISR:
                    self._ISR[agent_idx] = float('TimeLimit.truncated' not in infos[agent_idx])
                if agent_idx not in self._ep_length:
                    self._ep_length[agent_idx] = self._steps
        if all(done):
            not_tl_truncated = all(['TimeLimit.truncated' not in info for info in infos])

            for agent_idx in range(self.env.get_num_agents()):
                infos[agent_idx][self._group_name].update(CSR=float(not_tl_truncated))
                infos[agent_idx][self._group_name].update(ISR=self._ISR[agent_idx])
                infos[agent_idx][self._group_name].update(ep_length=self._ep_length[agent_idx])

            self._clear_stats()
        return obs, reward, done, infos

    def _clear_stats(self):
        self._ISR = {}
        self._ep_length = {}
        self._steps = 0
