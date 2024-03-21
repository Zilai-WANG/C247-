import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def train(model, optimizer, train_loader, epoch):
    
    # put the model into training mode
    model.train()
    
    # enumerate the train loader
    for i, data in enumerate(train_loader):
        
        # extract data from train loader (torch Variable)
        Xtrain = Variable(data[0])
        ytrain = Variable(data[1])
        
        # set optimizer gradient
        optimizer.zero_grad()
        
        # send input through model
        output = model(Xtrain)
        
        # calculate loss
        loss = F.cross_entropy(output, ytrain)
        
        # backprop
        loss.backward()
        
        # take a gradient step
        optimizer.step()
        
        # verbose
        if i % 10 == 0:
            print('Training Progress: \tEpoch {} [{}/{} ({:.2f}%)]\t\tLoss: {:.5f}'.format(
                epoch+1, i*len(Xtrain), len(train_loader.dataset), 100.*i/len(train_loader), loss.data))
    
    return model


def evaluate(model, data_loader, mode):
    
    # put the model into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    
    for i, data in enumerate(data_loader):
        
        # extract data from train loader (torch Variable)
        Xdata = Variable(data[0])
        ydata = Variable(data[1])
        
        # send input through model
        output = model(Xdata)
        
        # sum up batch loss
        test_loss += F.cross_entropy(output, ydata).data
        
        # find index of max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        
        # running sum of number of correct predictions
        correct += pred.eq(ydata.data.view_as(pred)).long().cpu().sum()

    # average test_loss
    test_loss /= len(data_loader.dataset)
    
    # verbose
    if mode == 'train':
        print('\tTrain loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    elif mode == 'val':
        print('\tValidation loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    elif mode == 'test':
        print('\tTest loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    else:
        pass
    
    return [test_loss, 1.*correct.item()/len(data_loader.dataset)]

import matplotlib.pyplot as plt

def train_and_evaluate(model, optimizer, data_loaders, num_epochs=10):
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    # Assuming test_loader is the third item in data_loaders if needed for evaluation
    # test_loader = data_loaders[2]

    # Initialize lists to store per-epoch loss and accuracy
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Variables to store performance metrics
        total_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train_preds += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        # Calculate average loss and accuracy over the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train_preds / total_train_samples
        train_loss_history.append(avg_train_loss)
        train_accuracy_history.append(train_accuracy)

        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_val_preds = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val_preds += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_preds / total_val_samples
        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Train Acc: {train_accuracy}, Val Loss: {avg_val_loss}, Val Acc: {val_accuracy}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_history, label='Train Accuracy')
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()





