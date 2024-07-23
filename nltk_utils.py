import numpy as np
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Télécharger les données nécessaires de NLTK
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Charger les données du fichier JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    return word_tokenize(sentence)

def stem(word):
    """
    Lemmatize the word to its root form.
    """
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Example:
    sentence = ["Bonjour", "comment", "vas", "tu"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Lemmatize each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

def predict_intent(sentence, words, classes, model):
    """
    Predict the intent of a given sentence.
    """
    tokenized_sentence = tokenize(sentence)
    bow = bag_of_words(tokenized_sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """
    Get the response for the predicted intent.
    """
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't find a suitable response."

# Exemple d'utilisation
if __name__ == "__main__":
    sentence = "Bonjour"
    # Construire la liste des mots et des classes
    words = [stem(w) for intent in intents['intents'] for pattern in intent['patterns'] for w in tokenize(pattern)]
    classes = [intent['tag'] for intent in intents['intents']]
    
    # Remplacer ce modèle par votre modèle entraîné réel
    class DummyModel:
        def predict(self, X):
            return np.array([[0.1, 0.9]])

    model = DummyModel()
    predicted_intents = predict_intent(sentence, words, classes, model)
    response = get_response(predicted_intents, intents)
    print(response)
