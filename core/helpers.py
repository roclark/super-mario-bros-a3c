import torch
from os.path import join
from .constants import PRETRAINED_MODELS
from .model import ActorCritic


class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end


def load_model(environment, model):
    model_name = join(PRETRAINED_MODELS, '%s.dat' % environment)
    model.load_state_dict(torch.load(model_name))
    return model


def initialize_model(env, environment, transfer):
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if transfer:
        model = load_model(environment, model)
    return model
