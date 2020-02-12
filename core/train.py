import torch
from .helpers import initialize_model
from .wrappers import wrap_environment
from torch.distributions import Categorical
from torch.nn.functional import log_softmax, softmax


def setup_process(rank, args):
    torch.manual_seed(123 + rank)
    env = wrap_environment(args.environment, args.action_space)
    model = initialize_model(env, args.environment, args.transfer)
    state = torch.from_numpy(env.reset())
    if not args.force_cpu:
        torch.cuda.manual_seed(123 + rank)
        model.cuda()
        state = state.cuda()
    model.train()
    return env, model, state


def train_loop(env, args, model, state, hx, cx):
    log_policies = []
    values = []
    rewards = []
    entropies = []

    while True:
        logits, value, hx, cx = model(state, hx, cx)
        policy = softmax(logits, dim=1)
        log_policy = log_softmax(logits, dim=1)
        entropy = -(policy * log_policy).sum(1, keepdim=True)

        multinomial = Categorical(policy)
        action = multinomial.sample().item()

        next_state, reward, done, _ = env.step(action)
        state = torch.from_numpy(next_state)
        if not args.force_cpu:
            state = state.cuda()
        values.append(value)
        log_policies.append(log_policy[0, action])
        rewards.append(reward)
        entropies.append(entropy)
        if done:
            state = torch.from_numpy(env.reset())
            if not args.force_cpu:
                state = state.cuda()
            return model, state, values, log_policies, rewards, entropies


def calculate_loss(args, loss_values):
    R = torch.zeros((1, 1), dtype=torch.float)
    gae = torch.zeros((1, 1), dtype=torch.float)
    if not args.force_cpu:
        R = R.cuda()
        gae = gae.cuda()
    actor_loss = 0
    critic_loss = 0
    entropy_loss = 0
    next_value = R

    for value, log_policy, reward, entropy in loss_values[::-1]:
        gae = gae * args.gamma * args.tau
        gae = gae + reward + args.gamma * next_value.detach() - value.detach()
        next_value = value
        actor_loss = actor_loss + log_policy * gae
        R = R * args.gamma + reward
        critic_loss = critic_loss + (R - value) ** 2 / 2
        entropy_loss = entropy_loss + entropy

    total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
    return total_loss


def update_network(optimizer, total_loss, model, global_model):
    optimizer.zero_grad()
    total_loss.backward()

    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            break
        global_param._grad = local_param.grad

    optimizer.step()
    return optimizer, model, global_model


def train(rank, global_model, optimizer, args):
    env, model, state = setup_process(rank, args)

    for episode in range(args.num_episodes):
        model.load_state_dict(global_model.state_dict())
        hx = torch.zeros((1, 512), dtype=torch.float)
        cx = torch.zeros((1, 512), dtype=torch.float)
        if not args.force_cpu:
            hx = hx.cuda()
            cx = cx.cuda()

        train_outputs = train_loop(env, args, model, state, hx, cx)
        model, state, values, log_policies, rewards, entropies = train_outputs

        loss_values = list(zip(values, log_policies, rewards, entropies))
        total_loss = calculate_loss(args, loss_values)
        optimizer, model, global_model = update_network(optimizer, total_loss,
                                                        model, global_model)
