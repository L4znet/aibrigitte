import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
nltk.download('punkt')
nltk.download('wordnet')

# initialise le lemmatizer
lemmatizer = WordNetLemmatizer()

# initialise les listes pour les mots, classes et documents
words = []
classes = []
documents = []
ignoreWords = ['?', '!', '.', ',', ';', ':', 'ta', 'ton', 'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'à', 'au', 'aux', 'et', 'ou', 'où', 'qui', 'que', 'quoi', 'pour', 'sur', 'dans', 'avec', 'sans', 'sous', 'par', 'après', 'avant', 'pendant', 'contre', 'en', 'vers', 'chez', 'hors', 'depuis', 'jusque', 'jusqu\'à', 'jusqu\'au']

# Charge le fichier JSON des intention 
dataFile = open('intents.json').read()
intents = json.loads(dataFile)

# Tokenise chaque mot dans les intention et creer les listes de mots, classes et documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenise chaque mot
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Ajoute les documents dans la collection de texte
        documents.append((w, intent['tag']))

        # Ajoute à notre liste de classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatise et mettre en minuscule chaque mot et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreWords]
words = sorted(list(set(words)))

# trie des classes
classes = sorted(list(set(classes)))

# Sauvegarder les mots et classes dans des fichiers pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Créer les données d'entrainement
training = []
# Créer un tableau vide pour notre sortie
outputEmpty = [0] * len(classes)

# Jeu d'entrainement, sac de mots pour chaque phrase
for doc in documents:
    # Initialiser notre sac de mots
    bag = []
    # Liste des mots tokenisés pour le modèle
    patternWords = doc[0]
    # Lemmatise chaque mot pour représenter les mots liés
    patternWords = [lemmatizer.lemmatize(word.lower()) for word in patternWords]
    # Créer notre tableau de sac de mots avec 1 si le mot est trouvé dans le modèle actuel
    for w in words:
        bag.append(1) if w in patternWords else bag.append(0)

    # La sortie est un '0' pour chaque tag et '1' pour le tag actuel (pour chaque modèle)
    outputRow = list(outputEmpty)
    outputRow[classes.index(doc[1])] = 1

    training.append([bag, outputRow])

# S'assurer que toutes les listes ont la même longueur
for i in range(len(training)):
    print(f"Length of bag {i}: {len(training[i][0])}")
    print(f"Length of output row {i}: {len(training[i][1])}")

# Mélanger nos caractéristiques et les transformer en np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Créer des listes d'entraînement et de test. X - modèles, Y - intentions
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Créer le modèle - 3 couches. Première couche 128 neurones, deuxième couche 64 neurones et la troisième couche de sortie contient le nombre de neurones
# égal au nombre d'intentions pour prédire l'intention de sortie avec softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile le modèle. La descente de gradient stochastique avec Nesterov accéléré donne de bons résultats pour ce modèle
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Ajustee et sauvegarder le modèle 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created and saved")


# lemmatise c'est la reduction de la variablité, ca met les mots a leur forme de base, ca simplifie l'analyse et le traitement
# les fichier pickle contiennent toute nos info en binaire, ce qui rend nos info plus compact 
# les bag of word (bow): c'est une technique pour representer des textes, ca convertie un doc textuel en vecteur de mot pour pouvoir appliquer des algo d'apprentissage automatique
# ^ represente des texte en chiffre 
# Stochastique = aleatoire
