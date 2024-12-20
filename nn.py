import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pretty_midi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from cs_senior_seminar import extract_features, abc_to_midi, read_abcs, convert_abc_to_midi

class MusicClassifierMLP(nn.Module):
    # a neural network class for classification task
    def __init__(self, input_dim, num_classes=9):
        super(MusicClassifierMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ComposerNN(nn.Module):
    # a neural network class for classification task
    def __init__(self, input_size, output_size):
        super(ComposerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        '''Function for forward pass'''
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
class MLPClassifier(nn.Module):
    # a neural network class for classification task
    def __init__(self, input_dim=128, num_classes=9, hidden_dim=256):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data(input_file):
    '''Get data from midi files and form composers and tunes dataset from it'''
    abc_tunes = read_abcs(input_file)
    midi_tunes = convert_abc_to_midi(abc_tunes)
    features = extract_features(midi_tunes)
    dataset, composers = make_dataset(features)
    return dataset, composers

def make_dataset(features):
    '''Make the dataset in appropriate format to be converted to dataloader'''
    composers = []
    dataset = []
    for composer, tunes in features.items():
        for tune in tunes:
            data = [tune['avg_pitch'],
                    tune['pitch_range'],
                    tune['pitch_sd'],
                    tune['avg_duration'],
                    tune['duration_range'],
                    tune['duration_sd'],
                    tune['avg_interval'],
                    tune['interval_range'],
                    tune['interval_sd']]
            dataset.append(data)
            composers.append(composer)
    dataset = np.array(dataset)
    scalar = StandardScaler()
    standardized_dataset = scalar.fit_transform(dataset)
                                   
    return standardized_dataset, composers

def prepare_data(dataset, composers):
    '''Prepaper dataset for the neural network'''
    Label_Encoder = LabelEncoder()
    labels = Label_Encoder.fit_transform(composers)

    # split train and val dataset into 80-20
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2) 

    # convert to tensor
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # create dataloaders
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_loader, test_loader


def run_model(model, train_loader, test_loader, num_epochs=500):
    '''Train the model, save the best model in the same directory, and plot train/val losses.'''

    # Cross entropy loss for classification task
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a high value

    # Construct the path to save the model in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    best_model_path = os.path.join(script_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        # Training loop
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {100*correct/total:.2f}%")

        # Validation loop
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        print(f"Test Loss: {avg_val_loss:.4f}, Test Accuracy: {100*correct/total:.2f}%")

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to '{best_model_path}' with validation loss: {best_val_loss:.4f}")

        model.train()  # Return to training mode

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    
def main():
    '''Main function to run the model'''
    input_file = 'abc.txt'
    dataset, composers = get_data(input_file)
    print(dataset)
    print(composers)
    train_loader, test_loader = prepare_data(dataset, composers)
    model = ComposerNN(train_loader.dataset.tensors[0].shape[1], len(set(composers)))
    
    #model = MLPClassifier(input_dim=train_loader.dataset.tensors[0].shape[1], num_classes=len(set(composers)))
    #model = MusicClassifierMLP(input_dim=train_loader.dataset.tensors[0].shape[1], num_classes=len(set(composers)))
    run_model(model, train_loader, test_loader)

if __name__ == '__main__':
    main()