
import matplotlib.pyplot as plt

import os
import torch
import argparse
# from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
from dataloader import ObservationActionDataset, ChunkedObservationActionDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


""" --- Model Definition --- """    
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


""" --- Training and Testing Function --- """
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (batch_idx % args.log_interval == 0) and args.verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] batch cum. Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx /  len(train_loader), loss.item()), end='\r', flush=True)
            if args.dry_run:
                break
    train_loss_avg = train_loss / len(train_loader.dataset)
    return train_loss_avg

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss


    test_loss_avg = test_loss / len(test_loader.dataset)
    return test_loss_avg

""" --- Main --- """
def main():
    # Training settings : load from arg parser
    parser = argparse.ArgumentParser(description='Supervised Learning for model base RL controller')
    parser.add_argument('--verbose', type=str, default=True,                    help='Verbose parameter to print training loss')
    parser.add_argument('--load-experiement', type=str, default='aliengo_model_based_base')
    parser.add_argument('--load-dataset', type=str, default='baseTaskMultActGood1')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',      help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',          help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',          help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',             help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,      help='For Saving the current Model')
    parser.add_argument('--model-name', type=str, default='model1',             help='Name For Saving the current Model')
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
    train_dataset = ObservationActionDataset('dataset/' + args.load_experiement + '/' + args.load_dataset + '/training_data.pt')
    try : 
        test_dataset  = ObservationActionDataset('dataset/' + args.load_experiement + '/' + args.load_dataset + '/testing_data.pt')
    except : 
        print('\nTesting done with same dataset as training...\n')
        test_dataset  = ObservationActionDataset('dataset/' + args.load_experiement + '/' + args.load_dataset + '/training_data.pt')

    print('\nLoaded dataset : ',args.load_dataset)
    print('In : ', 'dataset/' + args.load_experiement + '/' + args.load_dataset + '/training_data.pt')

    # Set dataset loader
    train_loader = DataLoader(train_dataset,**train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    # Retrieve input and output size
    input_size = train_dataset.observations.shape[-1]
    output_size = train_dataset.actions.shape[-1]    

    print('\nTraining Datapoints  : ', train_dataset.observations.shape[0]) 
    print('Testing  Datapoints  : ', train_dataset.observations.shape[0]) 
    print('\nInput  size : ',input_size)
    print('Output Size : ',output_size,'\n')

    # Define Model criteria : model, optimizer and loss criterion and scheduler
    model           = Model(input_size, output_size).to(device)
    optimizer       = optim.Adadelta(model.parameters(), lr=args.lr)
    train_criterion = nn.MSELoss() 
    test_criterion  = nn.MSELoss() 
    scheduler       = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_loss_list = []
    test_loss_list = []

    # train and evaluate the model
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, train_criterion)
        test_loss = test(model, device, test_loader, test_criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('\nTest Epoch: {} - Test set: Average loss: {:.4f}\n'.format(epoch, test_loss))
        scheduler.step()

    # Save the trained model
    if args.save_model:
        # Create logging directory if necessary
        logging_directory = f'model/{args.load_experiement}/{args.load_dataset}'
        if not os.path.exists(logging_directory):
            os.makedirs(logging_directory)
        torch.save(model.state_dict(),logging_directory + '/' + args.model_name + '.pt')
        print('\nModel saved as : ',logging_directory + '/' + args.model_name + '.pt\n')

    # Plots the results
    plt.plot(test_loss_list)
    # plt.plot(train_loss_list)
    plt.title('Loss on Testing Dataset')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()
    print('Everything went well, closing\n')