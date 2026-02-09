import argparse
import os
import uuid

import torch
import torchmetrics

from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

import wandb
import datetime

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    SemanticSegmentationTest,
    SemanticSegmentationTrain
)
from spectralwaste_segmentation import models

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, args: argparse.Namespace, output_folder: os.PathLike, experiment_name: str, suffix: str):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'experiment_name': experiment_name,
        'args': args,
    }
    torch.save(checkpoint, os.path.join(output_folder, f'{experiment_name}.{suffix}.pth'))

def median_frequency_exp(dataset: Dataset, shuffle: bool, batch_size: int, num_workers: int, num_classes: int, soft: float, device: str) -> torch.Tensor:
    # Process the dataset in parallel
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory_device=device)

    # Initialize counts
    classes_freqs = torch.zeros(num_classes, dtype=torch.int64, device=device)

    for _, target in loader:
        target = target.to(device)
        classes, counts = torch.unique(target, return_counts=True)
        classes = classes.to(device)
        counts = counts.to(device)
        ignore = torch.bitwise_or(classes < 0, classes >= num_classes).to(device)
        classes_freqs.index_add_(0, classes[~ignore], counts[~ignore]).to(device)

    zeros = classes_freqs == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training samples")

    result = classes_freqs.median().to(device) / classes_freqs.to(device)
    result[zeros] = 0  # avoid inf values
    return (result ** soft).to(device)

def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    model.train()
    mean_loss = torchmetrics.MeanMetric().to(device)
    class_iou = torchmetrics.JaccardIndex(num_classes=dataloader.dataset.num_classes, task='multiclass', average='none').to(device)

    for input, target in dataloader:
        if isinstance(input, list):
            input = [i.to(device) for i in input]
        else:
            input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)
        class_iou.update(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss.update(loss)

    lr_scheduler.step()
    mean_loss = mean_loss.compute()
    class_iou = class_iou.compute()
    miou = class_iou[1:].mean()
    iou_std = class_iou[1:].std()
    return mean_loss, class_iou, miou, iou_std

def evaluate(model, dataloader, criterion, device):
    model.eval()
    mean_loss = torchmetrics.MeanMetric().to(device)
    class_iou = torchmetrics.JaccardIndex(num_classes=dataloader.dataset.num_classes, task='multiclass', average='none').to(device)

    with torch.inference_mode() and torch.no_grad():
        for input, target in dataloader:
            if isinstance(input, list):
                input = [i.to(device) for i in input]
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)

            mean_loss.update(criterion(output, target))
            class_iou.update(output, target)

    mean_loss = mean_loss.compute()
    class_iou = class_iou.compute()
    miou = class_iou[1:].mean()
    iou_std = class_iou[1:].std()
    return mean_loss, class_iou, miou, iou_std

def main(args):
    print(f"Random seed: {args.random_seed}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Start epoch: {args.start_epoch}")
    print(f"Number of epochs: {args.max_epoch}")
    print(f"Model: {args.model}")
    print(f"Input mode: {args.input_mode}")
    print(f"Target mode: {args.target_mode}")
    print(f"Data path: {args.data_path}")
    print(f"Results path: {args.results_path}")
    print(f"Resume: {args.resume}")
    print(f"Test only: {args.test_only}")
    print(f"Wandb enable: {args.wandb_enable}")
    print(f"Wandb entity: {args.wandb_entity}")
    print(f"Wandb project: {args.wandb_project}")
    print(f"Train data shuffle: {args.train_data_shuffle}")
    print(f"Val data shuffle: {args.val_data_shuffle}")
    print(f"Test data shuffle: {args.test_data_shuffle}")
    experiment_name = f'{args.model}-{args.input_mode}-{args.target_mode}-{str(uuid.uuid4())[:4]}'.replace('_', '-').replace(',', '-')
    print(f"Experiment name: {experiment_name}")
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.mps.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    if ',' in args.input_mode:
        # Use multimodal dataset
        args.input_mode = args.input_mode.split(',')

    train_data = SpectralWasteSegmentation(args.data_path, random_seed=args.random_seed, deterministic=args.deterministic, split='train', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTrain(), target_type='semantic')
    val_data = SpectralWasteSegmentation(args.data_path, random_seed=args.random_seed, deterministic=args.deterministic, split='val', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='semantic')
    test_data = SpectralWasteSegmentation(args.data_path, random_seed=args.random_seed, deterministic=args.deterministic, split='test', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='semantic')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.train_data_shuffle, num_workers=args.num_workers, pin_memory_device=args.device)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.val_data_shuffle, num_workers=args.num_workers, pin_memory_device=args.device)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.test_data_shuffle, num_workers=args.num_workers, pin_memory_device=args.device)

    #print("train_data.num_channels type: ", type(train_data.num_channels))

    model = models.create_model(args.model, train_data.num_channels, train_data.num_classes).to(args.device)
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    optimizer, lr_scheduler = models.create_optimizers(args.model, model, args.max_epoch)

    # Calculate loss weights and define loss
    loss_weights = median_frequency_exp(train_data, args.train_data_shuffle, args.batch_size, args.num_workers, train_data.num_classes, 0.12, args.device)
    criterion = torch.nn.CrossEntropyLoss(loss_weights.to(args.device))

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if not args.test_only:
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        # Evaluate model
        test_loss, test_class_iou, test_miou, test_iou_std = evaluate(model, test_dataloader, criterion, args.device)
        print(f'test/loss: {test_loss} | test/class_iou: {test_class_iou.tolist()} | test/miou: {test_miou}, test/iou_std: {test_iou_std}')
        return

    experiment_start_time = datetime.datetime.now()

    # Train
    results_folder = os.path.join(args.results_path, experiment_start_time.strftime("%Y-%b-%d"), experiment_start_time.strftime("%H-%M-%S"))
    os.makedirs(results_folder, exist_ok=True)
    best_val_miou = 0.0

    wandb_run: wandb.Run = None

    # Start logging
    if args.wandb_enable:
        experiment_timestamp = experiment_start_time.strftime("%Y-%b-%d-%H-%M-%S")
        print(f'Wandb experiment name: {experiment_name}-{experiment_timestamp}')
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=f'{experiment_name}-{experiment_timestamp}', name=f'{experiment_name}-{experiment_timestamp}', config=args, job_type='train')

    for epoch in range(args.start_epoch, args.max_epoch):
        train_loss, train_class_iou, train_miou, train_iou_std = train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler, args.device)
        val_loss, val_class_iou, val_miou, val_iou_std = evaluate(model, val_dataloader, criterion, args.device)

        print(f'epoch: {epoch:04d} | seed: {args.random_seed} | learning-rate: {lr_scheduler.get_last_lr()[0]:.4f}')
        print(f'target-benchmark/mean-cross-entropy-loss-over-accuracy: {train_loss:.4f}')
        print(f'target-benchmark/miou: {train_miou:.4f}')
        print(f'target-benchmark/iou-standard-deviation: {train_iou_std:.4f}')
        for i in range(train_data.num_classes):
            print(f'target-benchmark/iou-{train_data.classes_names[i]}'.replace('_', '-') + f': {train_class_iou[i]:.4f}')
        print(f'evaluation-benchmark/mean-cross-entropy-loss-over-accuracy: {val_loss:.4f}')
        print(f'evaluation-benchmark/miou: {val_miou:.4f}')
        print(f'evaluation-benchmark/iou-standard-deviation: {val_iou_std:.4f}')
        for i in range(val_data.num_classes):
            print(f'evaluation-benchmark/iou-{val_data.classes_names[i]}'.replace('_', '-') + f': {val_class_iou[i]:.4f}')

        save_checkpoint(model, optimizer, lr_scheduler, epoch, args, results_folder, experiment_name, f'epoch-{epoch:04d}')

        if args.wandb_enable and wandb_run is not None:
            train_class_iou_dict = {f'run/target-benchmark/iou-{train_data.classes_names[i]}'.replace('_', '-'): train_class_iou[i] for i in range(train_data.num_classes)}
            val_class_iou_dict = {f'run/evaluation-benchmark/iou-{val_data.classes_names[i]}'.replace('_', '-'): val_class_iou[i] for i in range(val_data.num_classes)}
            optimizer_param_groups_lr_dict = {f'run/optimizer/param-groups-{i:04d}/learning-rate': optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))}
            optimizer_param_groups_weight_decay_dict = {f'run/optimizer/param-groups-{i:04d}/weight-decay': optimizer.param_groups[i]['weight_decay'] for i in range(len(optimizer.param_groups))}
            lr_scheduler_lr_dict = {f'run/learning-scheduler/learning-rate-{i:04d}': lr_scheduler.get_last_lr()[i] for i in range(len(lr_scheduler.get_last_lr()))}
            wandb_run.log({
                **optimizer_param_groups_lr_dict,
                **optimizer_param_groups_weight_decay_dict,
                **lr_scheduler_lr_dict,
                "run/target-benchmark/mean-cross-entropy-loss-over-accuracy": train_loss,
                "run/target-benchmark/miou": train_miou,
                "run/target-benchmark/iou-standard-deviation": train_iou_std,
                **train_class_iou_dict,
                "run/evaluation-benchmark/mean-cross-entropy-loss-over-accuracy": val_loss,
                "run/evaluation-benchmark/miou": val_miou,
                "run/evaluation-benchmark/iou-standard-deviation": val_iou_std, 
                **val_class_iou_dict
            }, step=epoch)
            # wandb_artifact = wandb.Artifact(name=f'{experiment_name}-{experiment_timestamp}-results', type='model')
            # wandb_artifact.add_file(local_path=os.path.join(results_folder, f'{experiment_name}.epoch-{epoch:04d}.pth'))
            # wandb_run.log_artifact(wandb_artifact)
            

        if val_miou > best_val_miou:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, results_folder, experiment_name, 'best')
            test_loss, test_class_iou, test_miou, test_iou_std = evaluate(model, test_dataloader, criterion, args.device)
            best_model = model
            best_val_miou = val_miou
            print(f'New best model at epoch: {epoch:04d} | seed: {args.random_seed} | train/loss: {train_loss:.4f} | val/loss: {val_loss:.4f} | val/miou: {val_miou:.4f}')
            if args.wandb_enable and wandb_run is not None:
                train_class_iou_dict = {f'best/target-benchmark/iou-{train_data.classes_names[i]}'.replace('_', '-'): train_class_iou[i] for i in range(train_data.num_classes)}
                val_class_iou_dict = {f'best/evaluation-benchmark/iou-{val_data.classes_names[i]}'.replace('_', '-'): val_class_iou[i] for i in range(val_data.num_classes)}
                test_class_iou_dict = {f'best/test-benchmark/iou-{test_data.classes_names[i]}'.replace('_', '-'): test_class_iou[i] for i in range(test_data.num_classes)}
                optimizer_param_groups_lr_dict = {f'best/optimizer/param-groups-{i:04d}/learning-rate': optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))}
                optimizer_param_groups_weight_decay_dict = {f'best/optimizer/param-groups-{i:04d}/weight-decay': optimizer.param_groups[i]['weight_decay'] for i in range(len(optimizer.param_groups))}
                lr_scheduler_lr_dict = {f'best/learning-scheduler/learning-rate-{i:04d}': lr_scheduler.get_last_lr()[i] for i in range(len(lr_scheduler.get_last_lr()))}
                wandb_run.log({
                    **optimizer_param_groups_lr_dict,
                    **optimizer_param_groups_weight_decay_dict,
                    **lr_scheduler_lr_dict,
                    "best/target-benchmark/mean-cross-entropy-loss-over-accuracy": train_loss,
                    "best/target-benchmark/miou": train_miou,
                    "best/target-benchmark/iou-standard-deviation": train_iou_std,
                    **train_class_iou_dict,
                    "best/evaluation-benchmark/mean-cross-entropy-loss-over-accuracy": val_loss,
                    "best/evaluation-benchmark/miou": val_miou,
                    "best/evaluation-benchmark/iou-standard-deviation": val_iou_std,
                    **val_class_iou_dict,
                    "best/test-benchmark/mean-cross-entropy-loss-over-accuracy": test_loss,
                    "best/test-benchmark/miou": test_miou,
                    "best/test-benchmark/iou-standard-deviation": test_iou_std,
                    **test_class_iou_dict
                }, step=epoch)
                wandb_artifact = wandb.Artifact(name=f'{experiment_name}-{experiment_timestamp}-results', type='model')
                wandb_artifact.add_file(local_path=os.path.join(results_folder, f'{experiment_name}.best.pth'))
                wandb_run.log_artifact(wandb_artifact)

        # if args.wandb_enable and wandb_run is not None:
        #     wandb_artifact = wandb.Artifact(name=f'{experiment_name}-{experiment_timestamp}-results', type='model')
        #     wandb_artifact.add_dir(local_path=results_folder)
        #     wandb_run.log_artifact(wandb_artifact)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--data-path', type=str, default='data/spectralwaste_segmentation')
    parser.add_argument('--results-path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    # setting
    parser.add_argument('--model', type=str, default='mininet')
    parser.add_argument('--input-mode', type=str, default='rgb')
    parser.add_argument('--target-mode', type=str, default='labels_rgb')
    parser.add_argument('--random-seed', type=int, default=random.randint(0, 1000000), help='An integer as the random seed (Default = random.randint(0, 1000000))')
    parser.add_argument('--train-data-shuffle', type=bool, default=True, help='Whether to shuffle the training data (Default = True)')
    parser.add_argument('--val-data-shuffle', type=bool, default=True, help='Whether to shuffle the validation data (Default = False)')
    parser.add_argument('--test-data-shuffle', type=bool, default=True, help='Whether to shuffle the test data (Default = True)')
    parser.add_argument('--deterministic', type=bool, default=False, help='Whether to use deterministic algorithms (Default = False)')
    # training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--resume', type=str, default='', help='Path of a checkpoint')
    parser.add_argument('--test-only', action='store_true')
    # wandb
    parser.add_argument('--wandb-enable', type=bool, default=True)
    parser.add_argument('--wandb-entity', type=str, default='russellstevenmelchiorre-british-columbia-institute-of-te')
    parser.add_argument('--wandb-project', type=str, default='cyrus-spectral-waste-segmentation')

    args = parser.parse_args()
    main(args)