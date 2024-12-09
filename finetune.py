import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re

# Create a custom Dataset for PyTorch DataLoader
class MusicDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def read_abcs(file):
    '''Read ABC file and return a list of ABC tunes with composer'''
    abc_tunes_with_composer = []

    with open(file, 'r') as f:
        abc_tunes = f.read().split('\n\n')

    # Process each ABC tune
    for abc_tune in abc_tunes:
        abc_tune = abc_tune.split('\n')

        try:
            composer_line = abc_tune[0].strip()
            composer = composer_line.split(":")[1].strip()  # Handle composer extraction
        except IndexError:
            print("Error: Composer information is missing or malformed.")
            continue

        note_lines = []

        for line in abc_tune:
            if not re.match(r"composer.*", line):
                if line.strip():
                    note_lines.append(line.strip())


        notes = " ".join(note_lines).replace("|", "").replace(":", "")
        
        abc_tunes_with_composer.append((notes, composer))

    return abc_tunes_with_composer

def parse_abc_content(content):
    '''Function to parse the content of ABC file'''
    composer_match = re.search(r"composer:\s*(.+)", content, re.IGNORECASE)
    composer = composer_match.group(1) if composer_match else "Unknown"

    note_lines = []
    for line in content.split("\n"):
        if not re.match(r"[A-Z]:", line):
            note_lines.append(line.strip())

    notes = " ".join(note_lines).replace("|", "").replace(":", "")
    return notes, composer

def tokenize_texts(texts):
    '''Function to tokenize the texts'''
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

import matplotlib.pyplot as plt

def train_model(train_loader, test_loader, model):
    '''Funtion to train the model'''
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_losses = []
    val_losses = []

    for epoch in range(10):  # Train for 3 epochs
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch

            # Pass data through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Compute the average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"avg train loss: {avg_train_loss} for epoch {epoch}")
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # Compute the average validation loss for the epoch
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Calculate accuracy for the epoch
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {accuracy*100:.2f}%")

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()


def make_data(data):
    # Encode composers as numerical labels
    texts, composers = zip(*data)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(composers)

    # Split into train/test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_inputs = tokenize_texts(train_texts)
    test_inputs = tokenize_texts(test_texts)
    train_data = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(train_labels))
    test_data = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(test_labels))

    return train_data, test_data, label_encoder

def main():
    filename = "sample_abc.txt"
    data = []
    notes_composer_data = read_abcs(filename)
    data.extend(notes_composer_data)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set([item[1] for item in data])))

    # Prepare the data for training
    train_data, test_data, label_encoder = make_data(data)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    #print(train_loader)

    # Train the model
    train_model(train_loader, test_loader, model)

if __name__ == '__main__':
    main()