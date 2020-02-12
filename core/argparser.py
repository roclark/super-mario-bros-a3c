from argparse import Action, ArgumentParser
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
from core.constants import (ACTION_SPACE,
                            BETA,
                            ENVIRONMENT,
                            GAMMA,
                            LEARNING_RATE,
                            MAX_ACTIONS,
                            NUM_EPISODES,
                            NUM_PROCESSES,
                            TAU)
from core.helpers import Range


ACTION_SPACE_CHOICES = {
    'right-only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT
}


class ActionSpace(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.choices.get(values, self.default))


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--action-space', action=ActionSpace,
                        choices=ACTION_SPACE_CHOICES, help='Specify the '
                        'action space to use as given by gym-super-mario-bros.'
                        ' Refer to the README for more details on the various '
                        'choices. Default: %s' % ACTION_SPACE,
                        default=ACTION_SPACE_CHOICES[ACTION_SPACE])
    parser.add_argument('--beta', type=float, help='The coefficient used in '
                        'the entropy calculation. Default: %s' % BETA,
                        default=BETA)
    parser.add_argument('--environment', type=str, help='The OpenAI gym '
                        'environment to use. Default: %s' % ENVIRONMENT,
                        default=ENVIRONMENT)
    parser.add_argument('--force-cpu', action='store_true', help='By default, '
                        'the program will run on the first supported GPU '
                        'identified by the system, if applicable. If a '
                        'supported GPU is installed, but all computations are '
                        'desired to run on the CPU only, specify this '
                        'parameter to ignore the GPUs. All actions will run '
                        'on the CPU if no supported GPUs are found. Default: '
                        'False')
    parser.add_argument('--gamma', type=float, help='Specify the discount '
                        'factor to use for rewards. Default: %s' % GAMMA,
                        default=GAMMA)
    parser.add_argument('--learning-rate', type=float, help='The learning rate'
                        ' to use. Default: %s' % LEARNING_RATE,
                        default=LEARNING_RATE)
    parser.add_argument('--max-actions', type=int, help='Specify the maximum '
                        'number of actions to repeat while in the testing '
                        'phase. Default: %s' % MAX_ACTIONS,
                        default=MAX_ACTIONS)
    parser.add_argument('--num-episodes', type=int, help='The number of '
                        'episodes to run in the given environment. Default: '
                        '%s' % NUM_EPISODES, default=NUM_EPISODES)
    parser.add_argument('--num-processes', type=int, help='The number of '
                        'training processes to run in parallel. Default: %s'
                        % NUM_PROCESSES, default=NUM_PROCESSES)
    parser.add_argument('--render', action='store_true', help='Specify to '
                        'render a visualization in another window of the '
                        'learning process. Note that a Desktop Environment is '
                        'required for visualization. Rendering scenes will '
                        'lower the learning speed. Default: False')
    parser.add_argument('--tau', type=float, help='The value used to calculate'
                        'GAE. Default: %s' % TAU, default=TAU)
    parser.add_argument('--transfer', action='store_true', help='Transfer '
                        'model weights from a previously-trained model to new '
                        'models for faster learning and improved accuracy. '
                        'Default: False')
    return parser.parse_args()
