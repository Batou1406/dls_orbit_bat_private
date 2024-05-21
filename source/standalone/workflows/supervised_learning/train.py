import torch
import argparse
# from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
from dataloader import ObservationActionDataset, ChunkedObservationActionDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


# Load the dataset
# dataset = ChunkedObservationActionDataset('dataset/aliengo_model_based_speed/mcQueenOne/training_data_chunk_*.pt')
dataset = ObservationActionDataset('dataset/aliengo_model_based_speed/mcQueenFour/training_data.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define model
class Model1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model1, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        x = F.elu(x)
        x = self.lin3(x)
        x = F.elu(x)
        x = self.lin4(x)
        return x

# Example model definition
input_size = dataset.observations.shape[-1]
output_size = dataset.actions.shape[-1]
model = Model(input_size, output_size).cuda()

print('Input  Size : ', dataset.observations.shape)
print('Output Size : ', dataset.actions.shape)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for obs_batch, action_batch in dataloader:
        obs_batch, action_batch = obs_batch.cuda(), action_batch.cuda()  # Move to GPU
        optimizer.zero_grad()
        outputs = model(obs_batch)
        loss = criterion(outputs, action_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


print('Training completed.')


def train(args, model, device, train_loader, optimizer, epoch, criteron):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    # Training settings : load from arg parser
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--load-dataset', type=str, default='dataset/aliengo_model_based_speed/mcQueenFour/training_data.pt')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',      help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',          help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',          help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',             help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,     help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seeds
    torch.manual_seed(args.seed)

    # Set device
    if use_cuda:  device = torch.device("cuda")
    else:         device = torch.device("cpu")

    # Set training and testing arguments
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    # Set dataset : train and test
    train_dataset = ObservationActionDataset(args.load_dataset)
    test_dataset  = ObservationActionDataset('dataset/aliengo_model_based_speed/mcQueenFour/training_data.pt')

    # Set dataset loader
    train_loader = DataLoader(train_dataset,**train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # Define Model criteria : model, optimizer and loss criterion and scheduler
    model           = Model(input_size, output_size).to(device)
    optimizer       = optim.Adadelta(model.parameters(), lr=args.lr)
    train_criterion = nn.MSELoss() 
    test_criterion  = nn.MSELoss() 
    scheduler       = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train and evaluate the model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_criterion)
        test(model, device, test_loader, test_criterion)
        scheduler.step()

    # Save the trained model
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
