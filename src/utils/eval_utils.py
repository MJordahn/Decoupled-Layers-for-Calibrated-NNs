import torchvision.transforms as transforms
import os
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN, CIFAR100
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError, Accuracy
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
from ood_metrics import fpr_at_95_tpr, auroc
import torch.nn as nn
import torch
from src.utils.metrics import *

SVHN_ROTATIONS = [10., 45., 90., 135., 180.]
CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'glass_blur', 'impulse_noise', 'jpeg_compression',
    'motion_blur', 'pixelate', 'shot_noise', 'snow'
]

CIFAR100_C_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur',
                'zoom_blur', 'snow', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate', 'jpeg_compression', 'speckle_noise',
                'gaussian_blur', 'spatter', 'saturate']

CIFAR10_C_PATH = "./data/CIFAR-10-C/"
CIFAR100_C_PATH = "./data/CIFAR100-C/"

def eval_train_data(model, dataset, batch_size, device, num_samples=1):
    train_dataset, num_classes, _, _ = get_dataset(dataset, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    y_preds = []
    y_targets = []
    with torch.no_grad():
        for j, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            if num_samples == 1:
                preds = model(images)
                probs = nn.functional.softmax(preds, dim=1)
                y_targets.append(labels)
                y_preds.append(probs)
            else:
                probs = model.forward_multisample(images, num_samples=num_samples)
                y_targets.append(labels)
                y_preds.append(probs)
    y_preds = torch.cat(y_preds, dim=0)
    y_targets = torch.cat(y_targets, dim=0)
    nll_value = nll(y_preds, y_targets) 
    return nll_value

def eval_test_data(model, dataset, batch_size, device, num_samples=1):
    test_dataset, num_classes, _, _ = get_dataset(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
    mce = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
    ece = ece.to(device)
    mce = mce.to(device)
    y_preds = []
    y_targets = []
    OOD_labels = []
    OOD_y_preds_logits = []
    with torch.no_grad():
        for j, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            if num_samples == 1:
                preds = model(images)
                probs = nn.functional.softmax(preds, dim=1)
            else:
                probs = model.forward_multisample(images, num_samples=num_samples)
            acc = accuracy(probs, labels)
            ece.update(probs, labels)
            mce.update(probs, labels)
            y_targets.append(labels)
            y_preds.append(probs)
            max_predictions = torch.max(probs.data, 1).values
            OOD_y_preds_logits.append(max_predictions)
            OOD_labels.append(torch.tensor([1]*len(labels)))
    y_preds = torch.cat(y_preds, dim=0)
    y_targets = torch.cat(y_targets, dim=0)
    ece_calc = ece.compute()
    mce_calc = mce.compute()
    acc = accuracy.compute()
    nll_value = nll(y_preds, y_targets) 
    brier_score = brier(y_preds, y_targets)
    return ece_calc, mce_calc, acc, nll_value, brier_score, OOD_y_preds_logits, OOD_labels

def get_dataset(dataset, train=False):
    if dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])
        if train==True:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="train")
        else:
            dataset = SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="test")
        num_classes = 10
        n_samples = 10000
        input_shape = [1, 32, 32]
    elif dataset == "CIFAR10":
        dataset = CIFAR10(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), train=train)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=10
    elif dataset == "CIFAR100":
        dataset = CIFAR100(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))]), train=train)
        input_shape=[1, 32, 32]
        n_samples=10000
        num_classes=100
    return dataset, num_classes, n_samples, input_shape

def eval_shift_data(model, dataset, batch_size, device, num_samples=1):
    if dataset == "SVHN":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)
    elif dataset == "CIFAR10":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)
    elif dataset == "CIFAR100":
        shift_dataset, num_classes, _, _ = get_dataset(dataset)
        shift_dataloader = DataLoader(shift_dataset, batch_size=batch_size)

    ece_overall = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
    mce_overall = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    ece_overall = ece_overall.to(device)
    mce_overall = mce_overall.to(device)
    corruption_ece_dict = {}
    corruption_mce_dict = {}
    if dataset == "SVHN":
        with torch.no_grad():
            for i, rotation in enumerate(SVHN_ROTATIONS):
                ece_rotation = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_rotation = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                for j, (images, labels) in enumerate(shift_dataloader):
                    images = images.to(device)
                    images = rotate(images, angle=rotation)
                    labels = labels.to(device)
                    if num_samples==1:
                        preds = model(images)
                        probs = nn.functional.softmax(preds, dim=1)
                    else:
                        probs = model.forward_multisample(images, num_samples=num_samples)
                    acc = accuracy(probs, labels)
                    ece_overall.update(probs, labels)
                    mce_overall.update(probs, labels)
                    ece_rotation.update(probs, labels)
                    mce_rotation.update(probs, labels)
                ece_rotation_calc = ece_rotation.compute()
                mce_rotation_calc = mce_rotation.compute()
                corruption_ece_dict[i+1] = ece_rotation_calc.to("cpu").numpy().tolist()*100
                corruption_mce_dict[i+1] = mce_rotation_calc.to("cpu").numpy().tolist()*100
    elif dataset == "CIFAR10":
        for corruption in CORRUPTIONS:
            data = np.load(CIFAR10_C_PATH + corruption + '.npy')
            targets = np.load(CIFAR10_C_PATH + 'labels.npy')
            for intensity in range(5):
                ece_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                shift_dataset.data = data[intensity*10000:(intensity+1)*10000]
                shift_dataset.targets = torch.LongTensor(targets[intensity*10000:(intensity+1)*10000])

                shift_dataloader = DataLoader(
                    shift_dataset,
                    batch_size=batch_size,
                    )
                with torch.no_grad():
                    for j, (images, labels) in enumerate(shift_dataloader):
                        images = images.to(device)
                        labels = labels.to(device)
                        if num_samples==1:
                            preds = model(images)
                            probs = nn.functional.softmax(preds, dim=1)
                        else:
                            probs = model.forward_multisample(images, num_samples=num_samples)
                        acc = accuracy(probs, labels)
                        ece_overall.update(probs, labels)
                        mce_overall.update(probs, labels)
                        ece_corruption.update(probs, labels)
                        mce_corruption.update(probs, labels)
                ece_corruption_calc = ece_corruption.compute()
                mce_corruption_calc = mce_corruption.compute()
                if intensity not in corruption_ece_dict.keys():
                    corruption_ece_dict[intensity] = ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] = mce_corruption_calc.to("cpu").numpy().tolist()*100
                else:
                    corruption_ece_dict[intensity] += ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] += mce_corruption_calc.to("cpu").numpy().tolist()*100
        for key in corruption_ece_dict.keys():
            corruption_ece_dict[key] /= len(CORRUPTIONS)
            corruption_mce_dict[key] /= len(CORRUPTIONS)
    elif dataset == "CIFAR100":
        for corruption in CIFAR100_C_CORRUPTIONS:
            data = np.load(CIFAR100_C_PATH + corruption + '.npy')
            targets = np.load(CIFAR100_C_PATH + 'labels.npy')
            for intensity in range(5):
                shift_dataset.data = data[intensity*10000:(intensity+1)*10000]
                shift_dataset.targets = torch.LongTensor(targets[intensity*10000:(intensity+1)*10000])
                ece_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='l1')
                mce_corruption = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm='max')
                shift_dataloader = DataLoader(
                    shift_dataset,
                    batch_size=batch_size,
                    )
                with torch.no_grad():
                    for j, (images, labels) in enumerate(shift_dataloader):
                        images = images.to(device)
                        labels = labels.to(device)
                        if num_samples==1:
                            preds = model(images)
                            probs = nn.functional.softmax(preds, dim=1)
                        else:
                            probs = model.forward_multisample(images, num_samples=num_samples)
                        acc = accuracy(probs, labels)
                        ece_overall.update(probs, labels)
                        mce_overall.update(probs, labels)
                        ece_corruption.update(probs, labels)
                        mce_corruption.update(probs, labels)
                ece_corruption_calc = ece_corruption.compute()
                mce_corruption_calc = mce_corruption.compute()
                if intensity not in corruption_ece_dict.keys():
                    corruption_ece_dict[intensity] = ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] = mce_corruption_calc.to("cpu").numpy().tolist()*100
                else:
                    corruption_ece_dict[intensity] += ece_corruption_calc.to("cpu").numpy().tolist()*100
                    corruption_mce_dict[intensity] += mce_corruption_calc.to("cpu").numpy().tolist()*100
        for key in corruption_ece_dict.keys():
            corruption_ece_dict[key] /= len(CORRUPTIONS)
            corruption_mce_dict[key] /= len(CORRUPTIONS)
    acc = accuracy.compute()
    ece_overall_calc = ece_overall.compute()
    mce_overall_calc = mce_overall.compute()

    return ece_overall_calc, mce_overall_calc, acc, corruption_ece_dict, corruption_mce_dict

def eval_ood_data(model, dataset, batch_size, device, OOD_y_preds_logits, OOD_labels, num_samples=1):
    ood_dataloaders = get_ood_datasets(dataset, batch_size)
    with torch.no_grad():
        for i, ood_test_dataloader in enumerate(ood_dataloaders):
            for j, (images, labels) in enumerate(ood_test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                if num_samples==1:
                    preds = model(images)
                    probs = nn.functional.softmax(preds, dim=1)
                else:
                    probs = model.forward_multisample(images, num_samples=num_samples)
                max_predictions = torch.max(probs.data, 1).values
                OOD_y_preds_logits.append(max_predictions)
                OOD_labels.append(torch.tensor([0]*len(labels)))

    OOD_labels = torch.cat(OOD_labels)
    OOD_y_preds_logits = torch.cat(OOD_y_preds_logits)
    auroc_calc = auroc(OOD_y_preds_logits.to("cpu").numpy().tolist(), OOD_labels.to("cpu").numpy().tolist()), 
    fpr95_calc = fpr_at_95_tpr(OOD_y_preds_logits.to("cpu").numpy().tolist(), OOD_labels.to("cpu").numpy().tolist())
    return auroc_calc, fpr95_calc
    
def get_ood_datasets(dataset, batch_size):
    if dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))])  
        ood_test_dataloaders = [
        DataLoader(CIFAR10(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size),
        DataLoader(CIFAR100(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size)
        ]
        return ood_test_dataloaders
    elif dataset == "CIFAR10":
        ood_test_dataloaders = [
            DataLoader(SVHN(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), split="test"), batch_size=batch_size),
            DataLoader(CIFAR100(os.getcwd()+"/data/", download=True, transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.247, 0.243, 0.261))]), train=False), batch_size=batch_size)
        ]
        return ood_test_dataloaders
    elif dataset == "CIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])
        ood_test_dataloaders = [
            DataLoader(SVHN(os.getcwd()+"/data/", download=True, transform=transform, split="test"), batch_size=batch_size),
            DataLoader(CIFAR10(os.getcwd()+"/data/", download=True, transform=transform, train=False), batch_size=batch_size)
        ]
        return ood_test_dataloaders