import torch
import torch.multiprocessing as _mp
from core.argparser import parse_args
from core.model import ActorCritic
from core.optimizer import GlobalAdam
from core.test import test
from core.train import train
from core.wrappers import wrap_environment
from os import environ


def main():
    torch.manual_seed(123)
    mp = _mp.get_context('spawn')
    args = parse_args()
    env = wrap_environment(args.environment, args.action_space)
    global_model = ActorCritic(env.observation_space.shape[0],
                               env.action_space.n)
    if not args.force_cpu:
        torch.cuda.manual_seed(123)
        global_model.cuda()
    global_model.share_memory()
    global_optimizer = GlobalAdam(global_model.parameters(),
                                  lr=args.learning_rate)
    processes = []

    for rank in range(args.num_processes):
        process = mp.Process(target=train,
                             args=(rank, global_model, global_optimizer, args))
        process.start()
        processes.append(process)
    process = mp.Process(target=test, args=(env, global_model, args))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()


if __name__ == '__main__':
    environ['OMP_NUM_THREADS'] = '1'
    main()
