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
        # pdb.set_trace()
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
        # pdb.set_trace()
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

        # Build reverse mapping from class labels to category names
        self.class_label_to_category_name = {v: k for k, v in self.targeter.items()}
        # Build mapping from category names to hierarchy categories
        self.category_name_to_hierarchy_category = {}
        # pdb.set_trace()
        for class_info in self.data_catg:
            category_name = class_info[category]
            hierarchy_category = class_info[hierarchy_level]
            self.category_name_to_hierarchy_category[category_name] = hierarchy_category
        # pdb.set_trace()

        # Build mapping from class labels to hierarchy categories
        self.class_label_to_hierarchy_category = {
            class_label: self.category_name_to_hierarchy_category[category_name]
            for class_label, category_name in self.class_label_to_category_name.items()
        }

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

        # Precompute other classes for each class within the same hierarchy category
        # Initialize an empty dictionary to hold other classes for each class
        self.other_classes = {}
        # pdb.set_trace()
        # Loop over each class label in the dataset
        for class_label in self.samples_by_class.keys():
            # Get the hierarchy category for the current class
            current_hierarchy_category = self.class_label_to_hierarchy_category[class_label]

            # Initialize a list to store other class labels in the same hierarchy category
            other_classes_list = []

            # Loop over all possible class labels
            for l in self.samples_by_class.keys():
                # Skip if it's the same class
                if l == class_label:
                    continue
                # Check if the class 'l' has the same hierarchy category as the current class
                if self.class_label_to_hierarchy_category[l] == current_hierarchy_category:
                    # Add the class label to the list
                    other_classes_list.append(l)

            # Convert the list to a NumPy array and assign it to the dictionary
            self.other_classes[class_label] = np.array(other_classes_list)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # pdb.set_trace()
        # current sample
        cur_path, cur_target = self.samples[index]

        # Select a random sample from the same class
        same_class_samples = self.samples_by_class[cur_target]
        same_class_path = np.random.choice(same_class_samples)
        # pdb.set_trace()
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


def create_dataset(data_dir, split, year, category, batch_size, k_categ, k=0, ):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
        dataset = INatDataset(data_dir, split=split, year=year, category=category, transform=transform)
    else:
        if k_categ == None:
            dataset = INatOKODataset(data_dir, split=split, year=year, category=category, k=k, transform=transform)
        else:
            dataset = INatOKODatasetHardK(data_dir, split=split, year=year, category=category, k=k, transform=transform,
                                          hierarchy_level=k_categ)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                              num_workers=8, drop_last=True, pin_memory=False, persistent_workers=False)

    # data_loader = create_loader(dataset, batch_size=batch_size, input_size=(3, 224, 224), use_prefetcher=False,
    # num_workers=2, distributed=False)
    return data_loader


if __name__ == '__main__':
    import time
    from tqdm import tqdm

    split = 'train'
    batch_size = 64
    data_dir = r'/scratch/ssd004/scratch/kkasa/data/inat_comp/2019/'
    k = 1
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = INatOKODataset(data_dir, split='train', year=2019, category='name', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                              num_workers=8, drop_last=True, pin_memory=False, persistent_workers=False)
    s1 = time.time()

    for images, labels in tqdm(dataset):
        start_time = time.time()  # Start the timer

        # Simulate some processing on the batch (optional)
        # time.sleep(0.1)  # You can remove this line, it's just for simulation

        end_time = time.time()  # End the timer
        batch_time = end_time - s1
        # batch_times.append(batch_time)
        print(f"Batch  took {batch_time} seconds to load")
        s1 = time.time()
    s2 = time.time()
    print(f'full dataset took: {s2 - s1}')
    import pdb;

    # pdb.set_trace()
