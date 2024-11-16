from collections import defaultdict
import os
import json
import pdb
import random

# from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import ImageFolder, default_loader
import torch
import numpy as np
from timm.data.loader import create_loader
from PIL import ImageFilter, ImageOps
from torchvision import transforms

from data.samplers import BalancedBatchSamplerV3, EfficientMinSamplesPerClassSampler


class INatDataset(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, loader=default_loader,
                 target_transform=None):
        # Initialize the ImageFolder parent class
        # super().__init__(os.path.join(root, f'train_val{year}'), )

        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        self.targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = self.data_catg[int(elem['category_id'])][category]
            if king not in self.targeter.keys():
                self.targeter[king] = indexer
                indexer += 1

        self.nb_classes = len(self.targeter)

        self.samples = []
        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

        # # Group samples by class
        # self.samples_by_class = defaultdict(list)
        # for path, label in self.samples:
        #     self.samples_by_class[label].append(path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # import pdb
        # regular training when k=0 or when testing
        # if (self.k == 0) or (self.split == 'val'):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class INatOKODataset(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, k=1, loader=default_loader,
                 target_transform=None):
        # Initialize the ImageFolder parent class
        # super().__init__(os.path.join(root, f'train_val{year}'), )

        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.transform = transform
        self.k = k
        self.loader = loader
        self.target_transform = target_transform

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        self.targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = self.data_catg[int(elem['category_id'])][category]
            if king not in self.targeter.keys():
                self.targeter[king] = indexer
                indexer += 1

        self.nb_classes = len(self.targeter)

        self.samples = []
        self.samples_by_class = {}

        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

            if target_current_true not in self.samples_by_class:
                self.samples_by_class[target_current_true] = []
            self.samples_by_class[target_current_true].append(path_current)

        # Precompute other classes for each class
        self.other_classes = {
            class_label: np.array([l for l in self.samples_by_class.keys() if l != class_label])
            for class_label in self.samples_by_class.keys()
        }
        # Group samples by class
        self.samples_by_class = defaultdict(list)
        for path, label in self.samples:
            self.samples_by_class[label].append(path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # current sample
        cur_path, cur_target = self.samples[index]

        # Select a random sample from the same class
        same_class_samples = self.samples_by_class[cur_target]
        same_class_path = np.random.choice(same_class_samples)

        # Select a random sample from a different class for the odd-k sample
        different_class_label = np.random.choice(self.other_classes[cur_target])

        different_class_samples = self.samples_by_class[different_class_label]
        different_class_path = np.random.choice(different_class_samples)

        # path, target = self.samples[index]
        cur_sample = self.loader(cur_path)
        if self.transform is not None:
            cur_sample = self.transform(cur_sample)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        images = [cur_sample]
        # targets = [cur_target]
        for path in [same_class_path, different_class_path]:
            # Load image using the default loader
            image = self.loader(path)

            # Apply transformation if specified
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)
            # targets.append(target)
        return torch.cat(images, dim=0), cur_target


class INatOKODatasetHardK(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, k=1, loader=default_loader,
                 target_transform=None, hierarchy_level='genus'):
        # Initialize the ImageFolder parent class
        # super().__init__(os.path.join(root, f'train_val{year}'), )

        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.transform = transform
        self.k = k
        self.loader = loader
        self.target_transform = target_transform
        self.hierarchy_level = hierarchy_level

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        self.targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = self.data_catg[int(elem['category_id'])][category]
            if king not in self.targeter.keys():
                self.targeter[king] = indexer
                indexer += 1

        self.nb_classes = len(self.targeter)

        self.samples = []
        self.samples_by_class = {}
        self.class_label_to_hierarchy_category = defaultdict(int)  # {species_id: hierarchy category}

        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

            if target_current_true not in self.samples_by_class:
                self.samples_by_class[target_current_true] = []
            self.samples_by_class[target_current_true].append(path_current)

            self.class_label_to_hierarchy_category[target_current] = categors[self.hierarchy_level]

        # Precompute other classes for each class within the same hierarchy category
        # Initialize an empty dictionary to hold other classes for each class
        self.other_classes = {}
        # Loop over each class label in the dataset
        for class_label in self.samples_by_class.keys():

            # Get the hierarchy category for the current class
            current_hierarchy_category = self.class_label_to_hierarchy_category[class_label]

            # Initialize a list to store other class labels in the same hierarchy category
            other_classes_list = []

            # Loop over all possible class labels
            for k_label in self.samples_by_class.keys():
                # Skip if it's the same class
                if k_label == class_label:
                    continue
                # Check if the class 'l' has the same hierarchy category as the current class
                if self.class_label_to_hierarchy_category[k_label] == current_hierarchy_category:
                    # Add the class label to the list
                    other_classes_list.append(k_label)

            # Convert the list to a NumPy array and assign it to the dictionary
            self.other_classes[class_label] = np.array(other_classes_list)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # current sample
        cur_path, cur_target = self.samples[index]

        # Select a random sample from the same class
        same_class_samples = self.samples_by_class[cur_target]
        same_class_path = np.random.choice(same_class_samples)
        # Select a random sample from a different class for the odd-k sample
        different_class_label = np.random.choice(self.other_classes[cur_target])

        different_class_samples = self.samples_by_class[different_class_label]
        different_class_path = np.random.choice(different_class_samples)
        # path, target = self.samples[index]
        cur_sample = self.loader(cur_path)
        if self.transform is not None:
            cur_sample = self.transform(cur_sample)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        images = [cur_sample]
        # targets = [cur_target]
        for path in [same_class_path, different_class_path]:
            # Load image using the default loader
            image = self.loader(path)

            # Apply transformation if specified
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)
            # targets.append(target)
        return torch.cat(images, dim=0), cur_target


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GrayScale(object):
    """
    GrayScale the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def create_dataset(data_dir, split, year, category, batch_size, k_categ, k=0, ):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([GrayScale(p=1.0),
                                 Solarization(p=1.0),
                                 GaussianBlur(p=1.0)]),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset = INatDataset(data_dir, split=split, year=year, category=category, transform=transform)

    # if split == 'train':
    #     labels = [label for _, label in dataset.samples]
    #     sampler = BalancedBatchSamplerV3(batch_size=batch_size, labels=labels)
    #     data_loader = torch.utils.data.DataLoader(dataset, num_workers=8, pin_memory=False,
    #                                               persistent_workers=False, batch_sampler=sampler)
    # else:
    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
    #                                               num_workers=8, drop_last=True, pin_memory=False,
    #                                               persistent_workers=False)

    if (k == 0) or (split != 'train'):
        if split == 'train':
            dataset = INatDataset(data_dir, split=split, year=year, category=category, transform=train_transform)
        else:
            dataset = INatDataset(data_dir, split=split, year=year, category=category, transform=test_transform)
    else:
        print(f'Odd-K hierarchy category: {k_categ}')
        if k_categ == None:
            dataset = INatOKODataset(data_dir, split=split, year=year, category=category, k=k,
                                     transform=train_transform)
        else:
            dataset = INatOKODatasetHardK(data_dir, split=split, year=year, category=category, k=k,
                                          transform=train_transform,
                                          hierarchy_level=k_categ)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                              num_workers=7, drop_last=True, pin_memory=False, persistent_workers=False)

    # data_loader = create_loader(dataset, batch_size=batch_size, input_size=(3, 224, 224), use_prefetcher=False,
    # num_workers=2, distributed=False)
    return data_loader
