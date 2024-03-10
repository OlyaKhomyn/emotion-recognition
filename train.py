import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from tqdm import tqdm

from consts import JSON_FILE
from dataset import get_loaders, EmotionDataset, CNNDataset, get_cnn_loaders
from model import EmotionRecModel, resnet


def get_cost_function():
    return nn.CrossEntropyLoss()


def get_optimizer(model, lr, wd):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def training_step(model, data_loader, optimizer, cost, device):
    model.train()
    cumulative_loss = 0.0
    acc_s = []
    samples = 0.0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        features, labels = batch

        features = features.to(device)
        labels = labels.to(device).flatten()

        pred = model(features)
        loss = cost(pred, labels)
        cumulative_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        samples += features.shape[0]

        with torch.no_grad():
            pred = nn.Softmax()(pred)
            acc = (torch.argmax(pred, 1) == labels).float().mean()
            acc_s.append(acc)

    return cumulative_loss/samples, sum(acc_s) / len(acc_s)


def test_step(model, data_loader, cost, device):
    model.eval()
    cumulative_loss = 0.0
    acc_s = []
    samples = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            features, labels = batch

            features = features.to(device)
            labels = labels.to(device).flatten()

            pred = model(features)
            loss = cost(pred, labels)
            cumulative_loss += loss.item()

            samples += features.shape[0]

            pred = nn.Softmax()(pred)
            acc = (torch.argmax(pred, 1) == labels).float().mean()
            acc_s.append(acc)

    return cumulative_loss/samples, sum(acc_s) / len(acc_s)


def train_model(batch_size=128, learning_rate=0.00001,
                weight_decay=0.005, epochs=200):
    device = 'cpu'
    data = EmotionDataset(JSON_FILE)
    train_loader, test_loader = get_loaders(data)

    model = EmotionRecModel().to(device)

    optimizer = get_optimizer(model, learning_rate, weight_decay)
    cost = get_cost_function()

    for e in range(epochs):
        train_cumm_loss, train_acc = training_step(model, train_loader, optimizer, cost, device)
        test_cumm_loss, test_acc = test_step(model, train_loader, cost, device)

        print('Epoch: {:d}'.format(e + 1))
        print('\tTraining loss {:.5f}, Training Accuracy {:.4f}'.format(train_cumm_loss, train_acc))
        print('\tValidation loss {:.5f}, Validation Accuracy {:.4f}'.format(test_cumm_loss, test_acc))

    torch.save(model.state_dict(), 'models/trained_sob.pt')


def get_cnn_optimizer(model, lr, wd):
    final_layer_weights = []

    for name, param in model.named_parameters():
        if name.startswith('fc'):
            final_layer_weights.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': final_layer_weights, 'lr': lr}
    ], weight_decay=wd)

    return optimizer


def train_cnn(batch_size=128, learning_rate=0.0001, weight_decay=0.0001, epochs=20):
    device = 'mps'
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size=(128, 128)),
        T.RandomRotation(degrees=25),
        T.RandomResizedCrop(128),
        T.RandomHorizontalFlip(),
        T.ColorJitter(hue=.05, saturation=.05)
    ])

    transformed_dataset = CNNDataset(JSON_FILE, transform)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size=(128, 128)),
    ])
    original_dataset = CNNDataset(JSON_FILE, transform)
    data = torch.utils.data.ConcatDataset([transformed_dataset, original_dataset])

    train_loader, test_loader = get_cnn_loaders(data)

    model = resnet().to(device)

    optimizer = get_cnn_optimizer(model, learning_rate, weight_decay)
    cost = get_cost_function()

    for e in range(epochs):
        train_cumm_loss, train_acc = training_step(model, train_loader, optimizer, cost, device)
        test_cumm_loss, test_acc = test_step(model, train_loader, cost, device)

        print('Epoch: {:d}'.format(e + 1))
        print('\tTraining loss {:.5f}, Training Accuracy {:.4f}'.format(train_cumm_loss, train_acc))
        print('\tValidation loss {:.5f}, Validation Accuracy {:.4f}'.format(test_cumm_loss, test_acc))

    torch.save(model.state_dict(), 'models/trained_cnn.pt')
