# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

import os
import sys
import time
import argparse
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from timm.data.transforms_factory import create_transform
from utils import ensure_dir, load_model, write_json


# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(train_size, test_size, data_dir, batch_size=1, num_workers=2, total_num=50000, image_size=224):
    # Setup the transformation
    my_transform = create_transform(
        input_size=(3, image_size, image_size),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=.9,
        interpolation='bicubic',
        is_training=False
    )

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=my_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=my_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    return train_loader, test_loader


# Test the model on clean dataset
def test(model, dataloader):
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total


# Load the log and generate the training line
def log_generation(result_dir):
    # Load the statistics in the log
    epochs, train_rate, test_rate = [], [], []
    log_path = os.path.join(result_dir, 'log.txt')
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        flag = 0
        for i in reader:
            if flag == 0:
                flag += 1
                continue
            else:
                epochs.append(int(i[0]))
                train_rate.append(float(i[1]))
                test_rate.append(float(i[2]))

    # Generate the success line
    plt.figure(num=0)
    plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    figure_path = os.path.join(result_dir, 'patch_attack_success_rate.png')
    plt.savefig(figure_path)
    plt.close(0)


# Initialize the patch
# TODO: Add circle type
def patch_initialization2(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch


def patch_initialization(patch_type='rectangle', patch_size=50):
    if patch_type == 'rectangle':
        patch_size = int(patch_size)
        patch = np.random.rand(3, patch_size, patch_size)
    return patch


# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location


# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                                + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total


def test_patch_verify(patch_type, target, patch, test_loader, model, device='cuda'):
    from utils import gen_mask
    from test_verified import verify_inference
    model.to(device)
    model.eval()
    dl = test_loader
    patch_width = patch.shape[1]
    mask_size = 16 * (int(patch_width / 16) + 2)
    img_mask, attn_mask = gen_mask(
        img_size=224,
        patch_size=16,
        mask_size=mask_size,
        with_zero_mask=True
    )
    img_mask, attn_mask = img_mask.to(device), attn_mask.to(device)
    use_attn_mask = True
    n_total, n_verified, n_correct, n_verified_correct = 0, 0, 0, 0
    start_time = time.time()
    num_samples = len(dl)
    for i, (img, label) in enumerate(dl):
        img = img.to(device)

        applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        adv_img = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                            + torch.mul((1 - mask.type(torch.FloatTensor)), img.type(torch.FloatTensor))
        adv_img = adv_img.to(device)

        majority_pred, verified, preds = verify_inference(model, adv_img, img_mask, attn_mask, use_attn_mask)
        target = label[0].item()
        correct = preds[0].cpu().item() == target
        n_total += 1
        if correct:
            n_correct += 1
        if verified:
            n_verified += 1
        if correct and verified:
            n_verified_correct += 1
        preds = '-' if verified and correct else preds.cpu().data.tolist()
        acc = n_correct * 100 / n_total
        verify_ratio = n_verified * 100 / n_total
        verified_acc = n_verified_correct * 100 / n_total
        print(f'{i + 1:6d}/{num_samples}; '
              f'correct {n_correct} ({acc:.3f}); '
              f'verified {n_verified} ({verify_ratio:.3f}); '
              f'verified_correct {n_verified_correct} ({verified_acc:.3f}); '
              f'preds {preds}; label {target}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
        'n_total': n_total,
        'n_verified': n_verified,
        'n_correct': n_correct,
        'n_verified_correct': n_verified_correct,
        'accuracy': n_correct * 100 / n_total,
        'verified_ratio': n_verified * 100 / n_total,
        'verified_accuracy': n_verified_correct * 100 / n_total,
        'accuracy_in_verified': n_verified_correct * 100 / n_verified,
        'elapsed_time': elapsed_time,
        'elapsed_time_per_sample': elapsed_time / n_total
    }


# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1.0, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch


def gen_adversarial_patch(model, data_dir, patch_size, target=859, epochs=20, lr=1.0, max_iteration=1000):
    patch_type = 'rectangle'
    probability_threshold = 0.9

    print('Generating adversarial patch')
    model.eval()

    # Load the datasets
    train_loader, test_loader = dataloader(1000, 1000, data_dir)

    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    patch = patch_initialization(patch_type, patch_size)

    best_patch_epoch, best_patch_success_rate = 0, 0
    best_patch = patch
    # Generate the patch
    for epoch in range(epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
                train_actual_total += 1
                applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, target,
                                                                probability_threshold, model, lr, max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if predicted[0].data.cpu().numpy() == target:
                    train_success += 1
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(patch_type, target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(patch_type, target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            best_patch = patch

    print("The best patch is found at epoch {} with success rate {}% on testset".format(
        best_patch_epoch, 100 * best_patch_success_rate))
    return best_patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test", help="experiment name")
    parser.add_argument("--model_arch", type=str, default="ViT_b16_224", help='model arch to use')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
    parser.add_argument('--test_size', type=int, default=1000, help="number of test images")
    # parser.add_argument('--noise_percentage', type=float, default=0.05, help="percentage of the patch size compared with the image size")
    parser.add_argument('--patch_size', type=int, default=50, help="size (width/height) of adversarial patch")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max number of iterations")
    parser.add_argument('--target', type=int, default=859, help="target label")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--data_dir', type=str, default='/datasets/imgNet/imagenet1k_valid_dataset/', help="dir of the dataset")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--output_dir', type=str, default='output', help='dir of the output')
    args = parser.parse_args()
    return args


def gen_patch(model, train_loader, test_loader, result_dir, args):
    pictures_dir = os.path.join(result_dir, 'training_pictures')
    ensure_dir(pictures_dir)
    log_path = os.path.join(result_dir, 'log.txt')

    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    patch = patch_initialization(args.patch_type, args.patch_size)
    print('The shape of the patch is', patch.shape)

    with open(log_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate = 0, 0
    best_patch = patch

    # Generate the patch
    for epoch in range(args.epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] != label and predicted[0].data.cpu().numpy() != args.target:
                train_actual_total += 1
                applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=(3, 224, 224))
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if predicted[0].data.cpu().numpy() == args.target:
                    train_success += 1
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig(os.path.join(pictures_dir, str(epoch) + " patch.png"))
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_success_rate, test_success_rate])

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            best_patch = patch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            plt.savefig(os.path.join(pictures_dir, "best_patch.png"))

        # Load the statistics and generate the line
        log_generation(result_dir)
    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))
    return best_patch


if __name__ == '__main__':
    args = parse_args()

    model = load_model(args.model_arch, 1000, None).cuda()
    model.eval()
    train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 50000)
    result_dir = os.path.join(args.output_dir, args.exp_name)
    ensure_dir(result_dir)

    best_patch = gen_patch(model, train_loader, test_loader, result_dir, args)
    verify_result = test_patch_verify(args.patch_type, args.target, best_patch, test_loader, model)
    write_json(verify_result, os.path.join(result_dir, 'result.json'))

