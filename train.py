import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk_utils import bag_of_words, tokenize, stem

# Charger les intentions
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Préparation des données
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Prétraitement des mots
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f'{len(xy)} patterns')
print(f'{len(tags)} tags:', tags)
print(f'{len(all_words)} unique stemmed words:', all_words)

# Création des données d'entraînement
X = []
y = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Division des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparamètres
num_epochs = 8000
batch_size = 8
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 64
output_size = len(tags)
print(f'Input size: {input_size}, Output size: {output_size}')

class ChatDataset(Dataset):
    def __init__(self, features, labels):
        self.n_samples = len(features)
        self.x_data = torch.tensor(features, dtype=torch.float32)
        self.y_data = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Créer les chargeurs de données
train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialiser le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Définir le modèle simplifié
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        return self.softmax(out)

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Définir la perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entraîner le modèle avec l'arrêt anticipé
best_val_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Passe avant
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Passe arrière et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation après chaque époque
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for (words, labels) in val_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Sauvegarde du modèle avec la meilleure perte de validation
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save({
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }, "best_model.pth")

    if (epoch+1) % 100 == 0:  # Afficher la perte toutes les 100 époques
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')

print(f'Training complete. Best model saved to best_model.pth')
