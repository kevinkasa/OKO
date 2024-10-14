from collections import defaultdict
import os
import json
import pdb
import tensorflow as tf
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch


class INatDataset(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, k=1):
        # Initialize the ImageFolder parent class
        super().__init__(os.path.join(root, f'train_val{year}'), transform=transform)

        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.k = k

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"{split}{year}.json")
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
        # regular training
        if self.k == 0:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            # Get the paths and labels for three consecutive images (including the current one)
            paths_and_labels = [self.samples[(index + i) % len(self.samples)] for i in range(3)]

            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            images = [sample]
            targets = [target]
            for path, target in paths_and_labels:
                # Load image using the default loader
                image = self.loader(path)

                # Apply transformation if specified
                if self.transform is not None:
                    image = self.transform(image)

                images.append(image)
                targets.append(target)

            return torch.cat(images, dim=0), targets[0]


def create_dataset(data_dir, split, year, category, batch_size, k=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = INatDataset(data_dir, split=split, year=year, category=category, k=k, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                              num_workers=16, drop_last=True)
    return data_loader
