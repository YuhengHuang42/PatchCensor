import os

import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, ImageNet
from torchvision.transforms import transforms
from timm.data.transforms_factory import create_transform
import numpy as np
import tarfile
from PIL import Image
from voter_study.PatchGuard.dataset_utils import PartImageNet

__all__ = ['MNISTDataLoader', 'CIFAR10DataLoader', 'ImageNetDataLoader', 'CIFAR100DataLoader']

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
                 batch_size=16, num_workers=8, shuffle=None, skip=None):
        if skip is not None:
            raise NotImplementedError
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
                 batch_size=16, num_workers=8, shuffle=None, skip=None):
        if split == 'train':
            train = True
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                RandomPatch(mask_size)
            ])
        else:
            train = False
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
                 special_data_flag=None, skip=None):
        self.skip = skip
        if split == 'train':
            transform = transforms.Compose([
                create_transform(
                    input_size=(3, image_size, image_size),
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
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
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
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
