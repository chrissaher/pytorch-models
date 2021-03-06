import os
import argparse
import torch
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
import torch.utils.tensorboard as tb
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vision_transformer import VisionTransformer


torch.manual_seed(0)

models = {
    'vit': VisionTransformer
}

# Some info about dataset (img_size, in_channels, num_classes)
dataset_info = {
    'cifar10': (32, 3, 10),
    'cifar100': (32, 3, 100),
    'mnist': (28, 1, 10),
}

datasets = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
}

def accuracy(pred, target):
    pred_argmax = pred.max(dim=-1)[1].type_as(target)
    return pred_argmax.eq(target).float().sum()

def load_data(dataset_name, root, num_workers=0, batch_size=1, **kwargs):
    ds = datasets[dataset_name](root, **kwargs)
    return DataLoader(ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def train(args):
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'valid'), flush_secs=1)

    assert args.model_name in models.keys()
    assert args.dataset_name in datasets.keys()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    attn_drop = args.attn_drop
    drop_rate = args.drop_rate
    patch_size = args.patch_size
    num_layers = args.num_layers
    mlp_hidden = args.mlp_hidden
    num_heads = args.num_heads

    ds_info = dataset_info[args.dataset_name]
    model = models[args.model_name](ds_info[0], ds_info[1], ds_info[2],
                                    patch_size=patch_size, attn_drop=attn_drop,
                                    mlp_hidden=mlp_hidden, drop_rate=drop_rate,
                                    num_layers=num_layers, num_heads=num_heads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_transfrom = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = load_data(args.dataset_name,
                           'data',
                           num_workers=0,
                           batch_size=batch_size,
                           train=True,
                           transform=train_transfrom,
                           download=True)


    valid_data = load_data(args.dataset_name,
                           'data',
                           num_workers=0,
                           batch_size=batch_size,
                           train=False,
                           transform=valid_transform,
                           download=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    train_index = 0
    for epoch in range(epochs):
        train_acc = 0
        valid_acc = 0
        train_loss = 0

        model.train()
        for input, target in tqdm(train_data):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(input)
            loss = criterion(pred, target)
            train_loss = round(loss.item(), 4)
            loss.backward()
            optimizer.step()
            if train_logger is not None:
                train_logger.add_scalar('loss', train_loss, train_index)
                train_acc += accuracy(pred, target).item()
            train_index += 1

        model.eval()
        for input, target in tqdm(valid_data):
            input, target = input.to(device), target.to(device)
            pred = model(input)
            valid_acc += accuracy(pred, target).item()

        train_acc = round(train_acc * 100 / len(train_data) / batch_size, 4)
        valid_acc = round(valid_acc * 100 / len(valid_data) / batch_size, 4)

        if train_logger is not None:
            train_logger.add_scalar('accuracy', train_acc, epoch)

        if valid_logger is not None:
            valid_logger.add_scalar('accuracy', valid_acc, epoch)

        if args.schedule_lr:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step(valid_acc)

        print(f"epoch: {epoch + 1} || train_loss: {train_loss} || train_acc: {train_acc} || valid_acc: {valid_acc}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epochs', type=int, default='10')
    parser.add_argument('-lr', '--learning_rate', type=float, default='0.01')
    parser.add_argument('-bs', '--batch_size', type=int, default='1')
    parser.add_argument('-slr', '--schedule_lr', action='store_true')
    parser.add_argument('-m', '--model_name', type=str, default='vit')
    parser.add_argument('-ds', '--dataset_name', type=str, default='cifar10')
    parser.add_argument('-r', '--root', type=str, default='./data')
    parser.add_argument('-ad', '--attn_drop', type=float, default=0.)
    parser.add_argument('-dr', '--drop_rate', type=float, default=0.)
    parser.add_argument('-ps', '--patch_size', type=int, default=8)
    parser.add_argument('-nl', '--num_layers', type=int, default=8)
    parser.add_argument('-mh', '--mlp_hidden', type=int, default=2048)
    parser.add_argument('-nh', '--num_heads', type=int, default=8)

    args = parser.parse_args()
    train(args)
