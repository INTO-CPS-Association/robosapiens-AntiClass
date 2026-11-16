import os
import sys
import json
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import *
from losses import *
from models import *

parser = argparse.ArgumentParser("AntiClass")
parser.add_argument('--task', type=str, default='classification', choices=['classification', 'open_set', 'ood'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use-gpu', type=str, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--val-interval', type=int, default=2)

parser.add_argument('--dataset', type=str, default='cifar100',
                    help='classification: mnist | cifar10 | cifar100, '
                         'open_set: mnist | svhn | cifar10 | cifar+10 | cifar+50 | tiny_imagenet, '
                         'ood: cifar10 | cifar100 (in-distribution dataset)')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--split', type=int, default=0, choices=range(5), help='known/unknown class split (open_set)')
parser.add_argument('--ood_datasets', nargs='+', default=None, choices=OOD_DATASETS,
                    help='OOD test sets (ood). Defaults to the near- and far-OOD sets of the OpenOOD benchmark.')
parser.add_argument('--checkpoint', type=str, default=None, help='evaluate this checkpoint instead of training')
parser.add_argument('--model', type=str, default='preactresnet18', help='model architecture',
                    choices=['preactresnet18', 'densenet121'])
parser.add_argument('--train_bs', type=int, default=128, help='batch size for trainloader')
parser.add_argument('--test_bs', type=int, default=256, help='batch size for testloader')
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.1, help='base learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--milestones', nargs='+', type=int, default=[50, 100])
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--weight-decay', type=float, default=1e-4, help="l2 regularization")

parser.add_argument('--w_ce', type=float, default=1.0)
parser.add_argument('--w_occe', type=float, default=1.0,
                    help="gamma parameter from the paper. We propose a full grid search in range (0.01, 1).")


def get_model(num_classes):
    if args.model == 'preactresnet18':
        model = preactresnet18(num_classes=num_classes)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=num_classes)
    else:
        raise ValueError("Model architecture not available.")

    if args.use_gpu:
        model.cuda(args.gpu)

    return model

def main():
    if args.task == 'ood' and args.dataset not in OOD_BENCHMARKS:
        raise ValueError(f"In-distribution dataset for ood must be one of {list(OOD_BENCHMARKS)}")

    outloader = None
    if args.task == 'open_set':
        dataset = OpenSetDataset(args.data_root, args=args)
        trainloader, testloader, outloader = dataset.get_loaders()
    else:
        dataset = ClosedSetDataset(args.data_root, train=True, args=args)
        trainloader = dataset.get_loader()

        test_dataset = ClosedSetDataset(args.data_root, train=False, args=args)
        testloader = test_dataset.get_loader()

    exp_name = f'{args.model}_ce={args.w_ce}_occe={args.w_occe}_e={args.n_epochs}_s={args.seed}'
    if args.task == 'open_set':
        out_dir = f'logs/open_set/{args.dataset}/{args.model}/{exp_name}_split={args.split}'
    elif args.task == 'ood':
        out_dir = f'logs/ood/{args.dataset}/{args.model}/' + exp_name
    else:
        out_dir = f'logs/{args.dataset}/{args.model}/' + exp_name
    writer = SummaryWriter(out_dir)

    occe_criterion = OCCELoss()
    ce_criterion = nn.CrossEntropyLoss()

    model = get_model(dataset.classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    save = args.checkpoint or f'{out_dir}/checkpoint.pth'
    if not args.checkpoint:
        train(model=model, ce_criterion=ce_criterion, occe_criterion=occe_criterion, optimizer=optimizer,
              scheduler=scheduler, trainloader=trainloader, testloader=testloader, writer=writer, save=save,
              outloader=outloader)
    if args.task == 'classification':
        return

    # final evaluation with the best checkpoint
    if os.path.exists(save):
        model.load_state_dict(torch.load(save, map_location='cpu'))
    if args.task == 'open_set':
        results = evaluate_open_set(model, testloader, outloader)
    else:
        results = evaluate_ood(model, testloader)
    for name, value in results.items():
        print(f'{name}: {value * 100:.2f}')
    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump({name: float(value) for name, value in results.items()}, f, indent=2)


def train(model, ce_criterion, occe_criterion, optimizer, scheduler, trainloader, testloader, writer, save,
          outloader=None):
    best_val_acc = float('-inf')
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        ce_losses, occe_losses, total_losses = AverageMeter(), AverageMeter(), AverageMeter()
        accuracy = AverageMeter()
        for data, labels in trainloader:
            if args.use_gpu:
                data, labels = data.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(data)

            loss_ce = ce_criterion(outputs, labels)
            loss_occe = occe_criterion(outputs, labels)
            total_loss = args.w_ce * loss_ce + args.w_occe * loss_occe

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch - 1) % args.val_interval == 0:
                with torch.no_grad():
                    _, pred = torch.max(outputs, 1)
                    accuracy.update((pred == labels).sum() / labels.size(0), labels.size(0))
                    ce_losses.update(loss_ce, labels.size(0))
                    occe_losses.update(loss_occe, labels.size(0))
                    total_losses.update(total_loss, labels.size(0))

        scheduler.step()
        if (epoch - 1) % args.val_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Train/Classifier_LR', current_lr, epoch)
            writer.add_scalar(f'Train/Accuracy', accuracy.avg, epoch)
            writer.add_scalar(f'Train/CE Loss', ce_losses.avg, epoch)
            writer.add_scalar(f'Train/OCCE Loss', occe_losses.avg, epoch)
            writer.add_scalar(f'Train/Total Loss', total_losses.avg, epoch)

            val_acc = evaluate(model, ce_criterion, occe_criterion, testloader, eval_idx=epoch, writer=writer)
            if outloader is not None:
                evaluate_open_set(model, testloader, outloader, eval_idx=epoch, writer=writer)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save), exist_ok=True)
                torch.save(model.state_dict(), save)


def evaluate(model, ce_criterion, occe_criterion, testloader, eval_idx, writer, saved_model=None):
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        if args.use_gpu:
            model.cuda(args.gpu)
    model.eval()
    total_losses, ce_losses, occe_losses = AverageMeter(), AverageMeter(), AverageMeter()
    accuracy = AverageMeter()
    with torch.no_grad():
        for data, labels in testloader:
            if args.use_gpu:
                data, labels = data.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(data)

            loss_ce = ce_criterion(outputs, labels)
            loss_occe = occe_criterion(outputs, labels)
            total_loss = args.w_ce * loss_ce + args.w_occe * loss_occe

            _, pred = torch.max(outputs, 1)
            accuracy.update((pred == labels).sum() / labels.size(0), labels.size(0))
            ce_losses.update(loss_ce, labels.size(0))
            occe_losses.update(loss_occe, labels.size(0))
            total_losses.update(total_loss, labels.size(0))

    writer.add_scalar(f'Val/Accuracy', accuracy.avg, eval_idx)
    writer.add_scalar(f'Val/CE Loss', ce_losses.avg, eval_idx)
    writer.add_scalar(f'Val/OCCE Loss', occe_losses.avg, eval_idx)
    writer.add_scalar(f'Val/Total Loss', total_losses.avg, eval_idx)

    return accuracy.avg


def get_logits(model, loader):
    logits, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            if args.use_gpu:
                data = data.cuda(args.gpu)
            logits.append(model(data).cpu())
            labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def evaluate_open_set(model, testloader, outloader, eval_idx=None, writer=None):
    model.eval()
    known_logits, labels = get_logits(model, testloader)
    unknown_logits, _ = get_logits(model, outloader)
    correct = (known_logits.argmax(dim=1) == labels).numpy()

    results = {'accuracy': correct.mean()}
    for score, score_fn in SCORES.items():
        known_scores, unknown_scores = score_fn(known_logits).numpy(), score_fn(unknown_logits).numpy()
        results[f'auroc_{score}'] = auroc(known_scores, unknown_scores)
        results[f'oscr_{score}'] = oscr(known_scores, unknown_scores, correct)

    if writer:
        for name, value in results.items():
            writer.add_scalar(f'OpenSet/{name}', value, eval_idx)
    return results


def evaluate_ood(model, testloader, eval_idx=None, writer=None):
    model.eval()
    benchmark = OOD_BENCHMARKS[args.dataset]
    ood_datasets = args.ood_datasets or benchmark['near'] + benchmark['far']
    if args.dataset in ood_datasets:
        raise ValueError("The in-distribution dataset cannot be used as an OOD dataset.")

    id_logits, _ = get_logits(model, testloader)
    results = {}
    for name in ood_datasets:
        ood_loader = get_ood_loader(name, args.data_root, batch_size=args.test_bs, num_workers=args.num_workers)
        ood_logits, _ = get_logits(model, ood_loader)
        for score, score_fn in SCORES.items():
            id_scores, ood_scores = score_fn(id_logits).numpy(), score_fn(ood_logits).numpy()
            results[f'{name}/auroc_{score}'] = auroc(id_scores, ood_scores)
            results[f'{name}/fpr95_{score}'] = fpr_at_95_tpr(id_scores, ood_scores)

    # average over the near- and far-OOD datasets that were evaluated
    for group in ['near', 'far']:
        names = [name for name in benchmark[group] if name in ood_datasets]
        if not names:
            continue
        for metric in [f'{m}_{score}' for m in ['auroc', 'fpr95'] for score in SCORES]:
            results[f'{group}_ood/{metric}'] = np.mean([results[f'{name}/{metric}'] for name in names])

    if writer:
        for name, value in results.items():
            writer.add_scalar(f'OOD/{name}', value, eval_idx)
    return results


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    if args.use_gpu:
        args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
    else:
        print("Currently using CPU")

    main()
