"""
Federated learning with different aggregation strategies on Office dataset
Extended with configurable noise injection for robustness experiments
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper import get_device
from utils.noise_utils import NoiseInjector, get_client_noise_config, print_noise_summary, create_mixed_domain_dataset, \
    NoisyDatasetWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils.data_utils import OfficeDataset
from nets.models import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total


################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def prepare_data(args):
    """
    Prepare data with optional noise injection and domain mixing
    Modified to support noise configuration per client
    """
    data_base_path = '../data'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
    ])

    # Load base datasets
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    # Create train/val splits
    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:])
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:])
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:])
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:])
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    # Store base datasets in list for easy access
    base_trainsets = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    base_valsets = [amazon_valset, caltech_valset, dslr_valset, webcam_valset]

    # Initialize noise injector with seed for reproducibility
    noise_injector = NoiseInjector(seed=args.noise_seed)

    # Get noise configurations for each client
    noise_configs = [get_client_noise_config(i, args) for i in range(4)]

    # Print noise summary
    datasets_names = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    print_noise_summary(datasets_names, noise_configs)

    # Apply domain mixing and noise
    final_trainsets = []
    final_valsets = []

    for client_idx in range(4):
        config = noise_configs[client_idx]

        # Handle domain mixing
        if config['mixed_domains'] is not None:
            mix_idx = config['mixed_domains']
            trainset = create_mixed_domain_dataset(
                base_trainsets[client_idx],
                base_trainsets[mix_idx]
            )
            valset = create_mixed_domain_dataset(
                base_valsets[client_idx],
                base_valsets[mix_idx]
            )
            print(f"Client {client_idx}: Mixed {datasets_names[client_idx]} + {datasets_names[mix_idx]}")
        else:
            trainset = base_trainsets[client_idx]
            valset = base_valsets[client_idx]

        # Apply noise wrapping
        if config['input_noise_ratio'] > 0 or config['label_noise_ratio'] > 0:
            trainset = NoisyDatasetWrapper(
                trainset,
                noise_injector,
                input_noise_ratio=config['input_noise_ratio'],
                label_noise_ratio=config['label_noise_ratio'],
                num_classes=31,
                gaussian_std=config['gaussian_std']
            )
            valset = NoisyDatasetWrapper(
                valset,
                noise_injector,
                input_noise_ratio=config['input_noise_ratio'],
                label_noise_ratio=config['label_noise_ratio'],
                num_classes=31,
                gaussian_std=config['gaussian_std']
            )

        final_trainsets.append(trainset)
        final_valsets.append(valset)

    # ============ END NEW CODE ============

    # Create data loaders
    train_loaders = [
        torch.utils.data.DataLoader(final_trainsets[0], batch_size=args.batch, shuffle=True),
        torch.utils.data.DataLoader(final_trainsets[1], batch_size=args.batch, shuffle=True),
        torch.utils.data.DataLoader(final_trainsets[2], batch_size=args.batch, shuffle=True),
        torch.utils.data.DataLoader(final_trainsets[3], batch_size=args.batch, shuffle=True)
    ]

    val_loaders = [
        torch.utils.data.DataLoader(final_valsets[0], batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(final_valsets[1], batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(final_valsets[2], batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(final_valsets[3], batch_size=args.batch, shuffle=False)
    ]

    # Test loaders remain clean (no noise)
    test_loaders = [
        torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False),
        torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)
    ]

    return train_loaders, val_loaders, test_loaders

if __name__ == '__main__':
    device = get_device()
    seed = 4
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--wandb', action='store_true', help='use weights and biases logging')
    parser.add_argument('--wandb_project', type=str, default='fedbn-office', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/office', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    # ============ Noise configuration arguments ============
    parser.add_argument('--noise_seed', type=int, default=42,
                       help='Random seed for noise injection (for reproducibility)')
    parser.add_argument('--gaussian_std', type=float, default=0.1,
                       help='Standard deviation for Gaussian input noise')

    # Method 1: Compact string format for all clients
    parser.add_argument('--client_noise', type=str, default='',
                       help='Client noise config: "0:input=0.2,label=0.0,mix=3;1:input=0.0,label=0.3,mix=2"')

    # Method 2: Individual client arguments (alternative, more verbose)
    parser.add_argument('--client0_input_noise', type=float, default=0.0,
                       help='Input noise ratio for client 0 (Amazon)')
    parser.add_argument('--client0_label_noise', type=float, default=0.0,
                       help='Label noise ratio for client 0 (Amazon)')
    parser.add_argument('--client0_mix_with', type=int, default=None,
                       help='Mix client 0 with another client (0-3)')

    parser.add_argument('--client1_input_noise', type=float, default=0.0,
                       help='Input noise ratio for client 1 (Caltech)')
    parser.add_argument('--client1_label_noise', type=float, default=0.0,
                       help='Label noise ratio for client 1 (Caltech)')
    parser.add_argument('--client1_mix_with', type=int, default=None,
                       help='Mix client 1 with another client (0-3)')

    parser.add_argument('--client2_input_noise', type=float, default=0.0,
                       help='Input noise ratio for client 2 (DSLR)')
    parser.add_argument('--client2_label_noise', type=float, default=0.0,
                       help='Label noise ratio for client 2 (DSLR)')
    parser.add_argument('--client2_mix_with', type=int, default=None,
                       help='Mix client 2 with another client (0-3)')

    parser.add_argument('--client3_input_noise', type=float, default=0.0,
                       help='Input noise ratio for client 3 (Webcam)')
    parser.add_argument('--client3_label_noise', type=float, default=0.0,
                       help='Label noise ratio for client 3 (Webcam)')
    parser.add_argument('--client3_mix_with', type=int, default=None,
                       help='Mix client 3 with another client (0-3)')
    # ============ END NEW ARGUMENTS ============

    args = parser.parse_args()

    if args.wandb:
        run_name = args.wandb_run_name if args.wandb_run_name else f"{args.mode}_lr{args.lr}_batch{args.batch}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "mode": args.mode,
                "learning_rate": args.lr,
                "batch_size": args.batch,
                "iterations": args.iters,
                "wk_iters": args.wk_iters,
                "mu": args.mu if args.mode.lower() == 'fedprox' else None,
                "noise_seed": args.noise_seed,
                "gaussian_std": args.gaussian_std,
                "client_noise": args.client_noise,
            }
        )

    exp_folder = 'fed_office'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    log = args.log
    if log:
        log_path = os.path.join('../logs/office/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    noise_seed: {}\n'.format(args.noise_seed))
        logfile.write('    client_noise: {}\n'.format(args.client_noise))

    train_loaders, val_loaders, test_loaders = prepare_data(args)

    # setup model
    server_model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()
    # name of each datasets
    datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    # federated client number
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/office/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_loss, train_acc = train_prox(args, model, train_loaders[client_idx],
                                                           optimizers[client_idx], loss_fun, device)
                    else:
                        train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx],
                                                      loss_fun, device)
                else:
                    train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun,
                                                  device)

                if args.wandb:
                    wandb.log({
                        f"train/loss_{datasets[client_idx]}": train_loss,
                        f"train/acc_{datasets[client_idx]}": train_acc,
                        "epoch": wi + a_iter * args.wk_iters,
                    })

        with torch.no_grad():
            # aggregation
            server_model, models = communication(args, server_model, models, client_weights)

            # Report loss after aggregation
            train_losses = []
            train_accs = []
            for client_idx, model in enumerate(models):
                train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                     train_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                                   train_loss,
                                                                                                   train_acc))

            if args.wandb:
                wandb.log({
                    "avg/train_loss": np.mean(train_losses),
                    "avg/train_acc": np.mean(train_accs),
                    "communication_round": a_iter,
                })

            # Validation
            val_acc_list = [None for j in range(client_num)]
            val_loss_list = []
            for client_idx, model in enumerate(models):
                val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
                val_acc_list[client_idx] = val_acc
                val_loss_list.append(val_loss)
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss,
                                                                                   val_acc), flush=True)
                if args.log:
                    logfile.write(
                        ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss,
                                                                                       val_acc))

                if args.wandb:
                    wandb.log({
                        f"val/loss_{datasets[client_idx]}": val_loss,
                        f"val/acc_{datasets[client_idx]}": val_acc,
                        "communication_round": a_iter,
                    })

            if args.wandb:
                wandb.log({
                    "avg/val_loss": np.mean(val_loss_list),
                    "avg/val_acc": np.mean(val_acc_list),
                    "communication_round": a_iter,
                })

            # Record best
            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed = True
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch,
                                                                                  best_acc[client_idx]))
                    if args.log:
                        logfile.write(
                            ' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch,
                                                                                       best_acc[client_idx]))

                if args.wandb:
                    wandb.log({
                        "best_avg_val_acc": np.mean(best_acc),
                        "best_epoch": best_epoch,
                    })

            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                if args.log:
                    logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() == 'fedbn':
                    torch.save({
                        'model_0': models[0].state_dict(),
                        'model_1': models[1].state_dict(),
                        'model_2': models[2].state_dict(),
                        'model_3': models[3].state_dict(),
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False

                    # Test on each client's test set
                    test_accs = []
                    test_losses = []
                    for client_idx, datasite in enumerate(datasets):
                        test_loss, test_acc = test(models[client_idx], test_loaders[client_idx], loss_fun, device)
                        test_accs.append(test_acc)
                        test_losses.append(test_loss)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                           test_acc))

                        if args.wandb:
                            wandb.log({
                                f"test/loss_{datasite}": test_loss,
                                f"test/acc_{datasite}": test_acc,
                                "communication_round": a_iter,
                            })

                    if args.wandb:
                        wandb.log({
                            "avg/test_loss": np.mean(test_losses),
                            "avg/test_acc": np.mean(test_accs),
                            "communication_round": a_iter,
                        })
                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False

                    # Test on each client's test set
                    test_accs = []
                    test_losses = []
                    for client_idx, datasite in enumerate(datasets):
                        test_loss, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device)
                        test_accs.append(test_acc)
                        test_losses.append(test_loss)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                           test_acc))

                        if args.wandb:
                            wandb.log({
                                f"test/loss_{datasite}": test_loss,
                                f"test/acc_{datasite}": test_acc,
                                "communication_round": a_iter,
                            })

                    if args.wandb:
                        wandb.log({
                            "avg/test_loss": np.mean(test_losses),
                            "avg/test_acc": np.mean(test_accs),
                            "communication_round": a_iter,
                        })

            if log:
                logfile.flush()

    if log:
        logfile.flush()
        logfile.close()

    if args.wandb:
        wandb.finish()