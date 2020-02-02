import torch
from collections import deque
from torch.nn.functional import softmax
from .model import ActorCritic
from .train_information import TrainInformation
from .wrappers import wrap_environment


def complete_episode(environment, info, episode_reward, episode, stats):
    new_best = info.update_rewards(episode_reward)
    if new_best and episode_reward >= info.best_reward:
        print('New high score of %s! Saving model' % round(episode_reward, 3))
    elif new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
    print('Episode %s - Reward: %s, Best: %s, Average: %s'
          % (episode,
             round(episode_reward, 3),
             round(info.best_reward, 3),
             round(info.average, 3)))


def test_loop(env, model, global_model, actions, state, done, args, info,
              episode_reward, hx, cx):
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
    if done or actions.count(actions[0]) == actions.maxlen:
        done = True
        info.update_index()
        complete_episode(args.environment, info, episode_reward, info.index,
                         stats)
        episode_reward = 0.0
        actions.clear()
        next_state = env.reset()
    state = torch.from_numpy(next_state)
    return model, hx, cx, state, done, info, episode_reward


def test(env, global_model, args):
    torch.manual_seed(123 + args.num_processes)
    info = TrainInformation()
    env = wrap_environment(args.environment, args.action_space)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()

    state = torch.from_numpy(env.reset())
    done = True
    episode_reward = 0.0
    step = 0
    hx = None
    cx = None
    actions = deque(maxlen=args.max_actions)

    while True:
        loop_outputs = test_loop(env, model, global_model, actions, state,
                                 done, args, info, episode_reward, hx, cx)
        model, hx, cx, state, done, info, episode_reward = loop_outputs
