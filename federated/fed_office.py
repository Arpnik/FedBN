"""
Federated learning with different aggregation strategies on Office dataset
Extended with configurable noise injection for robustness experiments
"""
import sys, os

from torch.utils.data import DataLoader
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper import get_device
from utils.data_utils import OfficeDataset, OfficeHomeDataset
from utils.noise_utils import get_client_noise_config, NoiseInjector, print_noise_summary, NoisyDatasetWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from nets.models import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np


def test_on_officehome(models, server_model, args, device, mode='fedbn',
                       client_idx=0, log_to_wandb=False, communication_round=None):
    """
    Test trained model(s) on Office-Home dataset for generalization evaluation

    Args:
        models: List of client models (for FedBN)
        server_model: Server model (for FedAvg/FedProx)
        args: Arguments containing officehome_path, batch size, etc.
        device: Computing device
        mode: Training mode ('fedbn', 'fedavg', 'fedprox')
        client_idx: Which client's model to use for testing (0-3)
        log_to_wandb: Whether to log results to wandb
        communication_round: Current communication round (for logging)

    Returns:
        dict: Dictionary with results for each Office-Home domain
    """

    # Check if Office-Home path is provided
    if not hasattr(args, 'officehome_path') or args.officehome_path is None:
        print("Office-Home path not provided, skipping Office-Home evaluation")
        return None

    if not os.path.exists(args.officehome_path):
        print(f"Office-Home path {args.officehome_path} does not exist, skipping evaluation")
        return None

    # Select model based on mode
    if mode.lower() == 'fedbn':
        if client_idx >= len(models):
            print(f"Warning: client_idx {client_idx} out of range, using client 0")
            client_idx = 0
        test_model = models[client_idx]
        model_name = f"Client_{client_idx}"
    else:
        test_model = server_model
        model_name = "Server"

    # Test transform (same as Office-Caltech test transform)
    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # Office-Home domains
    officehome_domains = ['Art', 'Clipart', 'Product', 'Real World']

    # Loss function
    loss_fun = nn.CrossEntropyLoss()

    # Store results
    results = {}
    all_losses = []
    all_accs = []

    print("\n" + "=" * 70)
    print(f"OFFICE-HOME EVALUATION ({model_name})")
    print("=" * 70)

    test_model.eval()
    with torch.no_grad():
        for domain in officehome_domains:
            try:
                # Load Office-Home domain
                officehome_dataset = OfficeHomeDataset(
                    args.officehome_path,
                    domain,
                    transform=transform_test
                )

                # Create data loader
                officehome_loader = DataLoader(
                    officehome_dataset,
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=2
                )

                # Test on this domain
                loss_all = 0
                total = 0
                correct = 0

                for data, target in officehome_loader:
                    data = data.to(device)
                    target = target.to(device)

                    output = test_model(data)
                    loss = loss_fun(output, target)

                    loss_all += loss.item()
                    total += target.size(0)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.view(-1)).sum().item()

                # Calculate metrics
                avg_loss = loss_all / len(officehome_loader)
                accuracy = correct / total

                # Store results
                results[domain] = {
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'total_samples': total
                }

                all_losses.append(avg_loss)
                all_accs.append(accuracy)

                # Print results
                print(f"  {domain:<12s} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Samples: {total}")

                # Log to wandb if enabled
                if log_to_wandb and wandb.run is not None:
                    log_dict = {
                        f"officehome/{domain}_loss": avg_loss,
                        f"officehome/{domain}_acc": accuracy,
                    }
                    if communication_round is not None:
                        log_dict["communication_round"] = communication_round
                    wandb.log(log_dict)

            except Exception as e:
                print(f"  {domain:<12s} | Error: {str(e)}")
                results[domain] = {'error': str(e)}

    # Calculate average across all domains
    if all_accs:
        avg_loss = np.mean(all_losses)
        avg_acc = np.mean(all_accs)

        results['average'] = {
            'loss': avg_loss,
            'accuracy': avg_acc
        }

        print(f"  {'Average':<12s} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        # Log average to wandb
        if log_to_wandb and wandb.run is not None:
            log_dict = {
                "officehome/avg_loss": avg_loss,
                "officehome/avg_acc": avg_acc,
            }
            if communication_round is not None:
                log_dict["communication_round"] = communication_round
            wandb.log(log_dict)

    print("=" * 70 + "\n")

    return results


def test_all_clients_on_officehome(models, server_model, args, device, mode='fedbn',
                                   log_to_wandb=False, communication_round=None):
    """
    Test all client models on Office-Home to compare generalization

    Args:
        models: List of client models
        server_model: Server model
        args: Arguments
        device: Computing device
        mode: Training mode
        log_to_wandb: Whether to log to wandb
        communication_round: Current round

    Returns:
        dict: Results for each client on each Office-Home domain
    """

    if mode.lower() != 'fedbn':
        # For FedAvg/FedProx, all clients use server model
        print("Testing server model on Office-Home...")
        return test_on_officehome(
            models, server_model, args, device, mode,
            client_idx=0, log_to_wandb=log_to_wandb,
            communication_round=communication_round
        )

    # For FedBN, test each client model
    all_results = {}

    print("\n" + "=" * 70)
    print("OFFICE-HOME EVALUATION - ALL CLIENTS")
    print("=" * 70)

    for client_idx in range(len(models)):
        print(f"\nTesting Client {client_idx}:")
        results = test_on_officehome(
            models, server_model, args, device, mode,
            client_idx=client_idx, log_to_wandb=False,  # Log manually below
            communication_round=communication_round
        )
        all_results[f'client_{client_idx}'] = results

        # Log to wandb with client prefix
        if log_to_wandb and wandb.run is not None and results:
            for domain, metrics in results.items():
                if 'error' not in metrics:
                    log_dict = {
                        f"officehome/client{client_idx}_{domain}_acc": metrics['accuracy'],
                    }
                    if communication_round is not None:
                        log_dict["communication_round"] = communication_round
                    wandb.log(log_dict)

    # Calculate best client per domain
    print("\nBest Client Performance per Domain:")
    print("-" * 70)

    officehome_domains = ['Art', 'Clipart', 'Product', 'RealWorld', 'average']
    for domain in officehome_domains:
        best_client = None
        best_acc = 0

        for client_idx in range(len(models)):
            client_key = f'client_{client_idx}'
            if client_key in all_results and domain in all_results[client_key]:
                if 'accuracy' in all_results[client_key][domain]:
                    acc = all_results[client_key][domain]['accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_client = client_idx

        if best_client is not None:
            print(f"  {domain:<12s} | Best: Client {best_client} with {best_acc:.4f}")

    print("=" * 70 + "\n")

    return all_results

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

def parse_client_datasets(client_datasets_str):
    """
    Parse client dataset assignment string.

    Args:
        client_datasets_str: String like "0:1:2:3" or "0,1:2:3"

    Returns:
        List of lists, e.g., [[0], [1], [2], [3]] or [[0,1], [2], [3]]
    """
    clients = client_datasets_str.split(':')
    return [[int(d) for d in client.split(',')] for client in clients]

def prepare_data_configurable(args):
    """
    Modified prepare_data function with configurable client-dataset assignment.
    Replace your existing prepare_data function with this.
    """
    data_base_path = '../data'
    transform_office = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # Load ALL base datasets
    all_datasets_train = [
        OfficeDataset(data_base_path, 'amazon', transform=transform_office),
        OfficeDataset(data_base_path, 'caltech', transform=transform_office),
        OfficeDataset(data_base_path, 'dslr', transform=transform_office),
        OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    ]

    all_datasets_test = [
        OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False),
        OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False),
        OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False),
        OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)
    ]

    # Parse client-dataset assignment
    client_dataset_assignment = parse_client_datasets(args.client_datasets)

    # Create train/val splits for all datasets
    min_data_len = min([len(ds) for ds in all_datasets_train])
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    all_trainsets = []
    all_valsets = []

    for ds in all_datasets_train:
        valset = torch.utils.data.Subset(ds, list(range(len(ds)))[-val_len:])
        trainset = torch.utils.data.Subset(ds, list(range(min_data_len)))
        all_trainsets.append(trainset)
        all_valsets.append(valset)

    # Initialize noise injector
    noise_injector = NoiseInjector(seed=args.noise_seed)
    datasets_names = ['Amazon', 'Caltech', 'DSLR', 'Webcam']

    # Prepare data for each client based on assignment
    final_trainsets = []
    final_valsets = []
    final_testsets = []
    client_names = []

    for client_idx, dataset_indices in enumerate(client_dataset_assignment):
        # Get noise config for this client
        config = get_client_noise_config(client_idx, args)

        # Combine datasets if multiple assigned to this client
        if len(dataset_indices) == 1:
            ds_idx = dataset_indices[0]
            trainset = all_trainsets[ds_idx]
            valset = all_valsets[ds_idx]
            testset = all_datasets_test[ds_idx]
            client_name = datasets_names[ds_idx]
        else:
            # Merge multiple datasets
            trainset = torch.utils.data.ConcatDataset([all_trainsets[i] for i in dataset_indices])
            valset = torch.utils.data.ConcatDataset([all_valsets[i] for i in dataset_indices])
            testset = torch.utils.data.ConcatDataset([all_datasets_test[i] for i in dataset_indices])
            client_name = '+'.join([datasets_names[i] for i in dataset_indices])

        # Apply noise wrapping
        if config['input_noise_ratio'] > 0 or config['label_noise_ratio'] > 0:
            trainset = NoisyDatasetWrapper(
                trainset, noise_injector,
                input_noise_ratio=config['input_noise_ratio'],
                label_noise_ratio=config['label_noise_ratio'],
                num_classes=31,
                gaussian_std=config['gaussian_std']
            )
            valset = NoisyDatasetWrapper(
                valset, noise_injector,
                input_noise_ratio=config['input_noise_ratio'],
                label_noise_ratio=config['label_noise_ratio'],
                num_classes=31,
                gaussian_std=config['gaussian_std']
            )

        final_trainsets.append(trainset)
        final_valsets.append(valset)
        final_testsets.append(testset)
        client_names.append(client_name)

    # Print configuration
    print("\n" + "=" * 70)
    print("CLIENT CONFIGURATION")
    print("=" * 70)
    for i, (name, indices) in enumerate(zip(client_names, client_dataset_assignment)):
        dataset_str = ', '.join([datasets_names[idx] for idx in indices])
        print(f"Client {i}: {name} (datasets: {dataset_str})")
    print("=" * 70 + "\n")

    # Create data loaders
    train_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
        for ds in final_trainsets
    ]

    val_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False)
        for ds in final_valsets
    ]

    test_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False)
        for ds in final_testsets
    ]

    return train_loaders, val_loaders, test_loaders, client_names


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
    parser.add_argument('--officehome_path', default='../data/OfficeHomeDataset_10072016', help='path ot office home dataset for testing')
    parser.add_argument('--officehome_test_client', type=int, default=0, help='0-3 for Amazon, Caltech, DSLR, Webcam respectively')


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
    parser.add_argument('--client_datasets', type=str,
                        default='0:1:2:3',
                        help='Dataset assignment for each client. Format: "0:1:2:3" or "0,1:2:3" for multi-dataset clients. '
                             'Datasets: 0=Amazon, 1=Caltech, 2=DSLR, 3=Webcam')

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
                "officehome_path": args.officehome_path,
                "officehome_test_client": args.officehome_test_client,
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

    train_loaders, val_loaders, test_loaders, datasets = prepare_data_configurable(args)

    # setup model
    server_model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()
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
                    checkpoint = {
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }
                    # Dynamically save all client models
                    for client_idx in range(client_num):
                        checkpoint[f'model_{client_idx}'] = models[client_idx].state_dict()
                    torch.save(checkpoint, SAVE_PATH)
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

    if args.officehome_path is not None:
        print("\n" + "="*70)
        print("Testing on Office-Home Dataset (Out-of-Distribution)")
        print("="*70)

        # For FedAvg/FedProx, use server model
        officehome_results = test_all_clients_on_officehome(
            models=models,
            server_model=server_model,
            args=args,
            device=device,
            mode=args.mode,
            log_to_wandb=args.wandb
        )

        if officehome_results and args.log:
            logfile.write("\n=== Office-Home Results ===\n")
            for domain, metrics in officehome_results.items():
                if 'accuracy' in metrics:
                    logfile.write(f"  {domain}: Acc={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}\n")

    if log:
        logfile.flush()
        logfile.close()

    if args.wandb:
        wandb.finish()