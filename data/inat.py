from collections import defaultdict
import os
import json
import pdb

# from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision import transforms
import torch
import numpy as np
from timm.data.loader import create_loader


class INatDataset(ImageFolder):
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
        # pdb.set_trace()
        # regular training when k=0 or when testing
        if (self.k == 0) or (self.split == 'val'):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            # current sample
            cur_path, cur_target = self.samples[index]

            # Select a random sample from the same class
            same_class_samples = self.samples_by_class[cur_target]
            same_class_path = np.random.choice(same_class_samples)

            # Select a random sample from a different class for the odd-k sample
            different_class_label = np.random.choice(self.other_classes[cur_target])

            different_class_samples = self.samples_by_class[different_class_label]
            different_class_path = np.random.choice(different_class_samples)

            # # Get the paths and labels for three consecutive images (including the current one)
            # paths_and_labels = [self.samples[(index + i) % len(self.samples)] for i in range(3)]

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


def create_dataset(data_dir, split, year, category, batch_size, k=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = INatDataset(data_dir, split=split, year=year, category=category, k=k, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                              num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)
    # data_loader = create_loader(dataset, batch_size=batch_size, input_size=(3, 224, 224), use_prefetcher=False,
                                # num_workers=2, distributed=False)
    return data_loader
