from copy import copy


class TrainInformation:
    def __init__(self):
        self._average = 0.0
        self._best_reward = -float('inf')
        self._best_average = -float('inf')
        self._rewards = []
        self._average_range = 100
        self._index = 0
        self._new_best_counter = 0
        self._saved_models = []

    @property
    def best_reward(self):
        return self._best_reward

    @property
    def best_average(self):
        return self._best_average

    @property
    def average(self):
        avg_range = self._average_range * -1
        return sum(self._rewards[avg_range:]) / len(self._rewards[avg_range:])

    @property
    def index(self):
        return self._index

    @property
    def new_best_counter(self):
        return self._new_best_counter

    def update_best_counter(self):
        self._new_best_counter += 1

    def _model_already_saved(self, new_model):
        for model in self._saved_models:
            if model == new_model:
                return True
        return False

    def store_actions(self, action_list):
        model_saved = self._model_already_saved(action_list)
        if not model_saved:
            # Create a copy of the action list to prevent it from being updated
            # elsewhere, making it appear that any new models are replicas of
            # existing ones, even when they are unique.
            copied_list = copy(action_list)
            self._saved_models.append(copied_list)
        # We want to update if the model has NOT been saved
        return not model_saved

    def _update_best_reward(self, episode_reward):
        if episode_reward > self.best_reward:
            self._best_reward = episode_reward
            return True
        return False

    def _update_best_average(self):
        if self.average > self.best_average:
            self._best_average = self.average
            return True
        return False

    def update_rewards(self, episode_reward):
        self._rewards.append(episode_reward)
        x = self._update_best_reward(episode_reward)
        y = self._update_best_average()
        if x or y:
            self.update_best_counter()
        return x or y

    def update_index(self):
        self._index += 1
