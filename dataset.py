import json

from torch.utils.data import Dataset
from feature_extraction import *
from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            js = f.read()
            self.data = json.loads(js)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]

        features = elem['features']

        label = torch.Tensor(1).long()
        label.data.fill_(elem['label'])

        return torch.tensor(features), label


class CNNDataset(Dataset):
    def __init__(self, data_path, transform):
        with open(data_path, 'r') as f:
            js = f.read()
            data = json.loads(js)
            neutral = [d for d in data if d['label'] == 4][:100]
            other = [d for d in data if d['label'] != 4]
            other.extend(neutral)
            self.data = other

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]

        im_path = elem['img']

        label = torch.Tensor(1).long()
        label.data.fill_(elem['label'])

        image = cv2.imread(im_path)
        image = self.transform(image)

        return image, label


def get_loaders(dataset):
    labels = [d['label'] for d in dataset.data]
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=test_sampler)

    return train_loader, test_loader


def get_cnn_loaders(dataset):
    labels = []
    for data in dataset.datasets:
        l = [d['label'] for d in data.data]
        labels.extend(l)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=test_sampler)

    return train_loader, test_loader
