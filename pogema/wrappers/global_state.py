from gym import ObservationWrapper


class GlobalStateWrapper(ObservationWrapper):
    def observation(self, observation):
        results = observation
        global_obstacles = self.grid.get_obstacles()
        global_agents_xy = self.grid.get_agents_xy()
        global_targets_xy = self.grid.get_targets_xy()

        for agent_idx in range(self.config.num_agents):
            result = results[agent_idx]
            result.update(global_obstacles=global_obstacles)
            result['global_xy'] = global_agents_xy[agent_idx]
            result['global_target_xy'] = global_targets_xy[agent_idx]

        return results

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)
