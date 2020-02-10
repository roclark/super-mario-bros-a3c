import torch
from collections import deque
from glob import glob
from gym.wrappers import Monitor
from os import listdir, mkdir, remove
from os.path import join
from shutil import move
from torch.nn.functional import softmax
from .constants import PRETRAINED_MODELS
from .model import ActorCritic
from .train_information import TrainInformation
from .wrappers import wrap_environment


def complete_episode(environment, info, episode_reward, episode, stats, model,
                     flag):
    new_best = info.update_rewards(episode_reward)
    save_model = False
    if new_best and episode_reward >= info.best_reward:
        print('New high score of %s! Saving model' % round(episode_reward, 3))
        save_model = True
    elif new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
    if flag:
        save_model = True
    if save_model:
        torch.save(model.state_dict(),
                   join(PRETRAINED_MODELS, '%s.dat' % environment))
    print('Episode %s - Reward: %s, Best: %s, Average: %s'
          % (episode,
             round(episode_reward, 3),
             round(info.best_reward, 3),
             round(info.average, 3)))


def test_loop(env, model, global_model, actions, state, done, args, info,
              episode_reward, hx, cx):
    flag = False
    if done:
        model.load_state_dict(global_model.state_dict())
    with torch.no_grad():
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
    logit, value, hx, cx = model(state, hx, cx)
    policy = softmax(logit, dim=1)
    action = torch.argmax(policy).item()
    next_state, reward, done, stats = env.step(action)
    if args.render:
        env.render()
    episode_reward += reward
    actions.append(action)
    if stats['flag_get']:
        print('Reached the flag!')
        flag = True
    if done or actions.count(actions[0]) == actions.maxlen:
        done = True
        info.update_index()
        complete_episode(args.environment, info, episode_reward, info.index,
                         stats, model, flag)
        actions.clear()
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True
        next_state = env.reset()
    state = torch.from_numpy(next_state)
    return model, hx, cx, state, done, info, episode_reward, flag


def determine_result(best_reward, flag, episode_reward):
    new_best = False
    if episode_reward > best_reward:
        best_reward = episode_reward
        new_best = True
    elif flag and episode_reward != best_reward:
        new_best = True
    return best_reward, new_best


class MonitorHandler:
    def __init__(self):
        self.new_best = True

    def video_callable(self, episode_id):
        return self.new_best


def test(env, global_model, args):
    torch.manual_seed(123 + args.num_processes)
    info = TrainInformation()
    env = wrap_environment(args.environment, args.action_space)
    handler = MonitorHandler()
    env = Monitor(env,
                  'recording/run1',
                  force=True,
                  video_callable=handler.video_callable)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()

    state = torch.from_numpy(env.reset())
    done = True
    episode_reward = 0.0
    hx = None
    cx = None
    actions = deque(maxlen=args.max_actions)
    best_reward = episode_reward

    while True:
        loop_outputs = test_loop(env, model, global_model, actions, state,
                                 done, args, info, episode_reward, hx, cx)
        model, hx, cx, state, done, info, episode_reward, flag = loop_outputs
        if done:
            best_reward, save_result = determine_result(best_reward, flag,
                                                        episode_reward)
            episode_reward = 0.0
            handler.new_best = save_result
