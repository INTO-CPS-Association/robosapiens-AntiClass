import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,  CenterCrop, \
    Resize, Grayscale


class ClosedSetDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None, args=None):
        self.train = train
        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        self.num_workers = getattr(args, 'num_workers', 0)
        if args.dataset == 'mnist':
            self.classes = 10
            self.transform = transform if transform else Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            self.dataset = datasets.MNIST(root=root, train=train, transform=self.transform, download=True)
        elif args.dataset == 'cifar10':
            self.classes = 10
            if self.train:
                self.transform = transform if transform else Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(32, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            else:
                self.transform = transform if transform else Compose([
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            self.dataset = datasets.CIFAR10(root=root, train=train, transform=self.transform, download=True)
        elif args.dataset == 'cifar100':
            self.classes = 100
            if self.train:
                self.transform = transform if transform else Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(32, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            else:
                self.transform = transform if transform else Compose([
                    CenterCrop(32),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            self.dataset = datasets.CIFAR100(root=root, train=train, transform=self.transform, download=True)
        else:
            raise ValueError("Dataset not available for this demo!")

    def get_loader(self):
        if self.train:
            return DataLoader(self.dataset, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers)
        else:
            return DataLoader(self.dataset, batch_size=self.test_bs, shuffle=False, num_workers=self.num_workers)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


# Known/unknown class splits for open-set recognition, as defined in ARPL
# (https://github.com/iCGY96/ARPL, MIT License). Five splits per benchmark.
OSR_SPLITS = {
    'mnist': [
        [2, 4, 5, 9, 8, 3],
        [3, 2, 6, 9, 4, 0],
        [5, 8, 3, 2, 4, 6],
        [3, 7, 8, 4, 0, 5],
        [6, 3, 4, 9, 8, 2]
    ],
    'svhn': [
        [5, 3, 7, 2, 8, 6],
        [3, 8, 7, 6, 2, 5],
        [8, 9, 4, 7, 2, 1],
        [3, 8, 2, 5, 0, 6],
        [4, 9, 2, 7, 1, 0]
    ],
    'cifar10': [
        [0, 6, 4, 9, 1, 7],
        [7, 6, 4, 9, 0, 1],
        [1, 5, 7, 3, 9, 4],
        [8, 6, 1, 9, 0, 7],
        [2, 4, 1, 7, 9, 6]
    ],
    # known CIFAR-10 classes for CIFAR+10 / CIFAR+50
    'cifar+': [
        [4, 7, 9, 1],
        [6, 7, 1, 9],
        [9, 6, 1, 7],
        [6, 4, 9, 1],
        [1, 0, 9, 8]
    ],
    # unknown CIFAR-100 classes for CIFAR+10
    'cifar+10': [
        [30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],
    # unknown CIFAR-100 classes for CIFAR+50
    'cifar+50': [
        [27, 94, 29, 77, 88, 26, 69, 48, 75, 5, 59, 93, 39, 57, 45, 40, 78, 20, 98, 47, 66, 70, 91, 76, 41, 83, 99, 32,
         53, 72, 2, 95, 21, 73, 84, 68, 35, 11, 55, 60, 30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [65, 97, 86, 24, 45, 67, 2, 3, 91, 98, 79, 29, 62, 82, 33, 76, 0, 35, 5, 16, 54, 11, 99, 52, 85, 1, 25, 66, 28,
         84, 23, 56, 75, 46, 21, 72, 55, 68, 8, 69, 41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [20, 83, 65, 97, 94, 2, 93, 16, 67, 29, 62, 33, 24, 98, 5, 86, 35, 54, 0, 91, 52, 66, 85, 84, 56, 11, 1, 76, 25,
         55, 21, 99, 72, 41, 23, 75, 28, 68, 69, 46, 8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [92, 82, 77, 64, 5, 33, 62, 56, 70, 0, 20, 28, 67, 14, 84, 53, 91, 29, 85, 2, 52, 83, 75, 35, 11, 21, 72, 98, 55,
         1, 41, 76, 25, 66, 69, 9, 48, 54, 40, 23, 95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [47, 6, 19, 0, 62, 93, 59, 65, 54, 70, 34, 55, 23, 38, 72, 76, 53, 31, 78, 96, 77, 27, 92, 18, 82, 50, 98, 32, 1,
         75, 83, 4, 51, 35, 80, 11, 74, 66, 36, 42, 33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],
    'tiny_imagenet': [
        [108, 147, 17, 58, 193, 123, 72, 144, 75, 167, 134, 14, 81, 171, 44, 197, 152, 66, 1, 133],
        [198, 161, 91, 59, 57, 134, 61, 184, 90, 35, 29, 23, 199, 38, 133, 19, 186, 18, 85, 67],
        [177, 0, 119, 26, 78, 80, 191, 46, 134, 92, 31, 152, 27, 60, 114, 50, 51, 133, 162, 93],
        [98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190],
        [95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170, 102, 177, 108, 169]
    ]
}
OSR_DATASETS = ['mnist', 'svhn', 'cifar10', 'cifar+10', 'cifar+50', 'tiny_imagenet']

# OpenOOD v1.5 benchmarks: near-OOD and far-OOD datasets for each in-distribution dataset
OOD_BENCHMARKS = {
    'cifar10': {'near': ['cifar100', 'tin'], 'far': ['mnist', 'svhn', 'texture', 'places365']},
    'cifar100': {'near': ['cifar10', 'tin'], 'far': ['mnist', 'svhn', 'texture', 'places365']},
}
OOD_DATASETS = ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365']

MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]


def get_transform(train, img_size=32, gray=False):
    if train:
        transform = [Resize((img_size, img_size)), RandomHorizontalFlip(),
                     RandomCrop(img_size, 4, padding_mode='reflect')]
    else:
        transform = [Resize(img_size), CenterCrop(img_size)]
    if gray:
        transform = [Grayscale(num_output_channels=3)] + transform
    return Compose(transform + [ToTensor(), Normalize(mean=MEAN, std=STD)])


class TinyImageNet(Dataset):
    """Tiny ImageNet with classes indexed by their sorted wnids. Downloaded to <root>/tiny-imagenet-200 if missing."""
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root='data', train=True, transform=None, download=True):
        base = os.path.join(root, 'tiny-imagenet-200')
        if not os.path.isdir(base) and download:
            download_and_extract_archive(self.url, root)
        with open(os.path.join(base, 'wnids.txt')) as f:
            wnids = sorted(line.strip() for line in f if line.strip())
        class_idx = {wnid: i for i, wnid in enumerate(wnids)}

        self.samples = []
        if train:
            for wnid in wnids:
                img_dir = os.path.join(base, 'train', wnid, 'images')
                self.samples += [(os.path.join(img_dir, name), class_idx[wnid]) for name in sorted(os.listdir(img_dir))]
        else:
            with open(os.path.join(base, 'val', 'val_annotations.txt')) as f:
                for line in f:
                    name, wnid = line.split('\t')[:2]
                    self.samples.append((os.path.join(base, 'val', 'images', name), class_idx[wnid]))
        self.targets = [target for _, target in self.samples]
        self.transform = transform

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


class ClassSubset(Dataset):
    """Keeps only the samples of the given classes and relabels them to 0..len(classes)-1."""

    def __init__(self, dataset, classes):
        targets = dataset.labels if hasattr(dataset, 'labels') else dataset.targets
        targets = np.asarray(targets)
        self.dataset = dataset
        self.indices = np.where(np.isin(targets, classes))[0]
        self.mapping = {c: i for i, c in enumerate(classes)}
        self.targets = [self.mapping[int(t)] for t in targets[self.indices]]

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        return img, self.targets[idx]

    def __len__(self):
        return len(self.indices)


def load_dataset(name, root, train, transform):
    if name == 'mnist':
        return datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif name == 'svhn':
        return datasets.SVHN(root=root, split='train' if train else 'test', transform=transform, download=True)
    elif name == 'cifar10':
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    elif name in ['tin', 'tiny_imagenet']:
        return TinyImageNet(root=root, train=train, transform=transform)
    elif name == 'texture':
        return ConcatDataset([datasets.DTD(root=root, split=split, transform=transform, download=True)
                              for split in ['train', 'val', 'test']])
    elif name == 'places365':
        return datasets.Places365(root=os.path.join(root, 'places365'), split='val', small=True,
                                  transform=transform, download=True)
    raise ValueError(f"Dataset {name} not available!")


class OpenSetDataset:
    """Open-set recognition benchmark: train/test on the known classes of a split, unknown samples for testing."""

    def __init__(self, root='data', args=None):
        if args.dataset not in OSR_DATASETS:
            raise ValueError(f"Open-set dataset must be one of {OSR_DATASETS}")
        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        self.num_workers = args.num_workers
        name, split = args.dataset, args.split
        img_size = 64 if name == 'tiny_imagenet' else 32
        gray = name == 'mnist'
        train_transform = get_transform(True, img_size, gray)
        test_transform = get_transform(False, img_size, gray)

        if name.startswith('cifar+'):
            known, unknown = OSR_SPLITS['cifar+'][split], OSR_SPLITS[name][split]
            self.train_set = ClassSubset(load_dataset('cifar10', root, True, train_transform), known)
            self.test_set = ClassSubset(load_dataset('cifar10', root, False, test_transform), known)
            self.out_set = ClassSubset(load_dataset('cifar100', root, False, test_transform), unknown)
        else:
            num_classes = 200 if name == 'tiny_imagenet' else 10
            known = OSR_SPLITS[name][split]
            unknown = [c for c in range(num_classes) if c not in known]
            self.train_set = ClassSubset(load_dataset(name, root, True, train_transform), known)
            test_set = load_dataset(name, root, False, test_transform)
            self.test_set = ClassSubset(test_set, known)
            self.out_set = ClassSubset(test_set, unknown)
        self.classes = len(known)
        print(f'{name} split {split} | known: {known} | train: {len(self.train_set)}, '
              f'test known: {len(self.test_set)}, test unknown: {len(self.out_set)}')

    def get_loaders(self):
        return (DataLoader(self.train_set, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers),
                DataLoader(self.test_set, batch_size=self.test_bs, shuffle=False, num_workers=self.num_workers),
                DataLoader(self.out_set, batch_size=self.test_bs, shuffle=False, num_workers=self.num_workers))


def get_ood_loader(name, root='data', batch_size=256, num_workers=4):
    """Test loader of an OOD dataset, resized to 32x32 and normalized like the in-distribution data."""
    if name not in OOD_DATASETS:
        raise ValueError(f"OOD dataset must be one of {OOD_DATASETS}")
    dataset = load_dataset(name, root, False, get_transform(False, 32, gray=name == 'mnist'))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
