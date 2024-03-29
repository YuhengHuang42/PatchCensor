import os

import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, ImageNet
from torchvision.transforms import transforms
from timm.data.transforms_factory import create_transform
import numpy as np
import tarfile
from PIL import Image
from voter_study.PatchGuard.dataset_utils import PartImageNet
import math

__all__ = ['MNISTDataLoader', 'CIFAR10DataLoader', 'ImageNetDataLoader', 'FOOD101DataLoader', 
           'CIFAR100DataLoader', 'GTSRBDataLoader', 'CelebADataLoader', 'FER2013DataLoader']

class ImageNet_File(Dataset):
    def __init__(self, imagenet_folder):
        self.imagenet_folder = imagenet_folder
    
    def __len__(self):
        return len(self.imagenet_folder)
    
    def __getitem__(self, idx):
        image_info = self.imagenet_folder[idx]
        file_name = self.imagenet_folder.samples[idx][0]
        return image_info[0], image_info[1], file_name
    
class RandomPatch(object):
    def __init__(self, mask_size=-1):
        super().__init__()
        self.mask_size = mask_size

    def __repr__(self):
        return self.__class__.__name__ + f'(mask_size={self.mask_size})'

    def __call__(self, img):
        """
            img (Tensor): Image to be transformed.
        Returns:
            Tensor: image with a random patch masked with 0.
        """
        ps = self.mask_size
        if ps <= 0:
            return img
        h, w = img.size(1), img.size(2)
        i = torch.randint(0, h - ps + 1, size=(1,)).item()
        j = torch.randint(0, w - ps + 1, size=(1,)).item()
        img[:, i:i + ps, j:j + ps] = 0.
        return img


class MNISTDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=28, batch_size=16, num_workers=1, shuffle=None, skip=None):
        if skip is not None:
            raise NotImplementedError
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])

        self.dataset = MNIST(root=data_dir, train=train, transform=transform, download=True)

        if shuffle is None:
            shuffle = False if not train else True
        super(MNISTDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)


class CIFAR10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None, 
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        if shuffle is None:
            shuffle = False if not train else True
        super(CIFAR10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)


class CIFAR100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None, model_type="vit"):
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        if shuffle is None:
            shuffle = False if not train else True
        super(CIFAR100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, 
                 mask_size=-1, batch_size=16, num_workers=8, shuffle=None, 
                 special_data_flag=None, skip=None, model_type="vit"):
        self.skip = skip
        if model_type == "vit":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        if split == 'train':
            transform = transforms.Compose([
                create_transform(
                    input_size=(3, image_size, image_size),
                    mean=mean,
                    std=std,
                    crop_pct=.9,
                    interpolation='bicubic',
                    is_training=True
                ),
                RandomPatch(mask_size)
            ])
            is_valid_file = None
        else:
            transform = transforms.Compose([
                create_transform(
                    input_size=(3, image_size, image_size),
                    mean=mean,
                    std=std,
                    crop_pct=.9,
                    interpolation='bicubic',
                    is_training=False
                ),
                RandomPatch(mask_size)
            ])

            def is_valid_file(fpath):
                fname = os.path.basename(fpath)
                return True if fname.startswith('ILSVRC2012_val_') and fname.endswith('.JPEG') else False

        if data_dir.endswith('_tar'):
            self.dataset = ImageTarDataset(
                tar_file=os.path.join(data_dir, f'{split}.tar'),
                return_labels=True,
                transform=transform
            )
        elif special_data_flag == "PartImageNet":
            dataset_ = ImageFolder(os.path.join(data_dir, split),
                                   transform=transform)
            self.dataset = PartImageNet(dataset_)
        else:
            self.dataset = ImageFolder(
                root=os.path.join(data_dir, split),
                transform=transform,
                is_valid_file=is_valid_file
            )
            self.dataset = ImageNet_File(self.dataset)
            if self.skip is not None:
                skips = list(range(0, len(self.dataset), self.skip))
                self.dataset = torch.utils.data.Subset(self.dataset, skips)
        print(f'loaded ImageNet {split} dataset. len={len(self.dataset)}')

        if shuffle is None:
            shuffle = True if split == 'train' else False
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)


class ImageTarDataset(Dataset):
    def __init__(self, tar_file, return_labels=False, transform=None):
        '''
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        '''
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, 'r:') as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count('/') != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(tar_file, self.num))
        self.return_labels = return_labels
        self.transform = transform

    def _get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')

        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert('RGB')
        image = self.transform(image)

        if self.return_labels:
            category = self.categories[self._get_category_from_filename(
                self.tar_members[index].name)]
            return image, category
        return image


# # How to use the ImageTarDataset class
# if __name__ == '__main__':
#
#     # path to the imagenet validation dataset, both options would work
#     imagenet_valid = os.path.join(os.environ['AMLT_DATA_DIR'], 'imagenet_valid.tar')
#     # imagenet_valid = '/mnt/data/imagenet/imagenet_valid.tar'
#
#     # load dataset
#     print('Loading dataset with various transforms')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])
#     dataset = ImageTarDataset(imagenet_valid, return_labels=True, transform=transform)
#     print('Building loader')
#     loader = DataLoader(dataset)
#     for x in loader:
#         # print first image
#         print(x)
#         break


if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='../data/imagenet2012',
        split='validation',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)

    
class FGVCAircraftDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None,
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            resize_size = image_size + 32
            transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        if split == "validation":
            split = "val"
        self.transform = transform
        self.dataset = torchvision.datasets.FGVCAircraft(root=data_dir, split=split, transform=transform, download=True)

        if shuffle is None:
            shuffle = False if not train else True
        super(FGVCAircraftDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
def split_trainset(train_proportion, split,
                        dataset, index_path):
    dataset_length = len(dataset)
    idx_path = os.path.join(index_path, "len_{}.pt".format(dataset_length))
    if os.path.isfile(idx_path):
        indices = torch.load(idx_path)
    else:
        indices = torch.randperm(dataset_length)
        torch.save(indices, idx_path)
    train_length = math.ceil(train_proportion * dataset_length)
    eval_proportion = 1 - train_proportion
    eval_length = math.ceil(eval_proportion * dataset_length)
    if split == "train":
        train_indices = indices[:train_length]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        return train_dataset
    else:
        eval_indices = indices[train_length: ] 
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
        return eval_dataset
    
class GTSRBDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None,
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            resize_size = image_size + 32
            transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        self.transform = transform
        if split == "validation" or split == "train":
            # no val in GTSRB
            self.dataset = torchvision.datasets.GTSRB(root=data_dir, split="train", transform=transform, download=True)
            self.dataset = split_trainset(0.8, split, self.dataset, data_dir)
        else:
            self.dataset = torchvision.datasets.GTSRB(root=data_dir, split=split, transform=transform, download=True)
        
        if shuffle is None:
            shuffle = False if not train else True
        super(GTSRBDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

class CelebADataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None,
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            resize_size = image_size + 32
            transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        if split == "validation":
            split = "valid" # no val in GTSRB
        self.transform = transform
        self.dataset = torchvision.datasets.CelebA(root=data_dir, split=split,
                                                   download=True, target_type="identity", 
                                                   transform=transform, target_transform=self._normalize_label)

        if shuffle is None:
            shuffle = False if not train else True
        super(CelebADataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        
    def _normalize_label(self, label):
        return label - 1

class FOOD101DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None,
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            resize_size = image_size + 32
            transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        self.transform = transform
        if split == "validation" or split == "train":
            # no val in Food101
            self.dataset = torchvision.datasets.Food101(root=data_dir, split="train", transform=transform, download=True)
            self.dataset = split_trainset(0.9, split, self.dataset, data_dir)
        else:
            self.dataset = torchvision.datasets.Food101(root=data_dir, split=split, transform=transform, download=True)
        
        if shuffle is None:
            shuffle = False if not train else True
        super(FOOD101DataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

class FER2013DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, mask_size=-1, 
                 batch_size=16, num_workers=8, shuffle=None, skip=None,
                 special_data_flag=None, model_type="vit"):
        if skip is not None:
            raise NotImplementedError
        if model_type == "vit":
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            train = True
            resize_size = image_size + 32
            transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
                RandomPatch(mask_size)
            ])
        self.transform = transform
        if split == "validation" or split == "train":
            # no val in GTSRB
            self.dataset = torchvision.datasets.FER2013(root=data_dir, split="train", transform=transform)
            self.dataset = split_trainset(0.8, split, self.dataset, data_dir)
        else:
            self.dataset = torchvision.datasets.FER2013(root=data_dir, split=split, transform=transform)
        
        if shuffle is None:
            shuffle = False if not train else True
        super(FER2013DataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)