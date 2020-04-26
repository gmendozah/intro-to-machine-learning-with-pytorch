import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json

# simple example
# python train.py "flowers" --arch "vgg16" --gpu
# complete example
# python train.py "flowers" --save_dir "train_checkpoint.pth" --arch "vgg16" --learning_rate 0.002 --hidden_units 512  --epochs 2 --gpu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='train_checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_model(arch, hidden_units, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        num_in_features = 1024

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)
    return model, device, num_in_features


def train_model(epochs, trainloader, validateloader, model, device, criterion, optimizer):
    steps = 0
    running_loss = 0
    print_every = 5

    start = time.time()
    print('Model is Training...')

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)  # Move input and label tensors to the default device

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validateloader:
                        inputs, labels = inputs.to(device), labels.to(device)  # transfering tensors to the GPU

                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(validateloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(validateloader):.3f}")
                running_loss = 0
                model.train()

    end = time.time()
    total_time = end - start
    print(" Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))


def save_checkpoint(file_path, model, image_datasets, epochs, optimizer, learning_rate, input_size, output_size, arch, hidden_units):
    model.class_to_idx = image_datasets[0].class_to_idx
    bundle = {
        'pretrained_model': arch,
        'input_size': input_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    torch.save(bundle, file_path)
    print("Model saved to storage...")

def main():
    print("Loading the data...")
    args = parse_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_data = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testing_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    validation_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = [
        datasets.ImageFolder(train_dir, transform=training_data),
        datasets.ImageFolder(valid_dir, transform=validation_data),
        datasets.ImageFolder(test_dir, transform=testing_data)
    ]

    dataloaders = [
        torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)
    ]

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # getting model, device object and number of input features
    model, device, num_in_features = load_model(args.arch, args.hidden_units, args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(args.epochs, dataloaders[0], dataloaders[1], model, device, criterion, optimizer)

    file_path = args.save_dir

    output_size = 102
    save_checkpoint(file_path, model, image_datasets, args.epochs, optimizer, args.learning_rate,
                    num_in_features, output_size, args.arch, args.hidden_units)


if __name__ == "__main__":
    main()