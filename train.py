import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.environments import PairedComparison
from src.model import GRU


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='From PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training')

    parser.add_argument('--epochs', type=int, default=10000, metavar='E', help='number of epochs to train')
    parser.add_argument('--warmup-epochs', type=int, default=5000, metavar='WE', help='number of epochs to warmup')
    parser.add_argument('--num-steps', type=int, default=100, metavar='N', help='number of batches in one epochs')

    parser.add_argument('--num-points', type=int, default=10, metavar='NS', help='number of query points')

    parser.add_argument('--num-cues', type=int, default=4, metavar='NE', help='number of cues')
    parser.add_argument('--num-hidden', type=int, default=128, metavar='NE', help='number of hidden units')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')
    parser.add_argument('--alpha', type=float, default=0, metavar='A', help='kl factor')
    parser.add_argument('--sampling', action='store_true', default=False, help='uses sampling')

    parser.add_argument('--direction', action='store_true', default=False, help='uses directed data-sets')
    parser.add_argument('--ranking', action='store_true', default=False, help='sort data-set according to importance')

    parser.add_argument('--num-runs', type=int, default=1, metavar='NR', help='number of runs')
    parser.add_argument('--save-path', default='trained_models/darmstadt_pre_', help='directory to save results')
    parser.add_argument('--load-path', default='trained_models/default_model_0.pth', help='path to load model')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--var', type=float, default=0.01, metavar='V', help='Standard deviation of the noise in the data')
    parser.add_argument('--theta', type=float, default=1.0, metavar='TH', help='Scale/Variance of Features')
    parser.add_argument('--eta', type=float, default=2.0, metavar='ET', help='Covariance of Features. Big eta more gaussian, small eta higher correlation')
    parser.add_argument('--w-std', type=float, default=1.0, metavar='WS', help='Standard deviation of ground truth weights.')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for i in range(args.num_runs):
        writer = SummaryWriter()
        performance = torch.zeros(args.epochs)
        accuracy_test = 0

        data_loader = PairedComparison(num_inputs=args.num_cues, num_targets=1, direction=args.direction, ranking=args.ranking, dichotomized=False, var=args.var, theta=args.theta, cov_prior_eta=args.eta, weight_std=args.w_std)
        model = GRU(data_loader.num_inputs, data_loader.num_targets, args.num_hidden).to(device)
        if args.alpha > 0:
            print('Loading pretrained network...')
            params, _ = torch.load(args.load_path)
            model.load_state_dict(params)
            model.reset_log_sigma()
            max_alpha = args.alpha
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

        with trange(args.epochs) as t:
            for j in t:
                loss_train = 0
                for k in range(args.num_steps):
                    inputs, targets, _, _ = data_loader.get_batch(args.batch_size, args.num_points, device=device)
                    predictive_distribution, _, _ = model(inputs, targets, args.sampling)

                    loss = -predictive_distribution.log_prob(targets).mean()
                    writer.add_scalar('NLL', loss.item(), j*args.num_steps + k)

                    if args.alpha > 0:
                        alpha = min(j / args.warmup_epochs, 1.0) * max_alpha
                        kld = model.regularization(alpha)
                        loss = loss + kld
                        writer.add_scalar('KLD', kld.item(), j*args.num_steps + k)

                    loss_train += loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
                    optimizer.step()

                t.set_description('Loss (train): {:5.4f}'.format(loss_train.item() / args.num_steps))
                performance[j] = loss_train.item() / args.num_steps

        torch.save([model.state_dict(), performance], args.save_path + str(i) + '.pth')

if __name__ == '__main__':
    main()
