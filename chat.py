import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def cleanUpSentence(sentence):
     # Tokenise la phrase - séparer les mots dans un tableau
    sentenceWords = nltk.word_tokenize(sentence)
    # Lemmatizer chaque mot
    sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
    return sentenceWords

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words):
     # Tokeniser la phrase
    sentenceWords = cleanUpSentence(sentence)
    # Sac de mots, matrice de vocabulaire
    bag = [0]*len(words)  
    for s in sentenceWords:
        for i,w in enumerate(words):
            if w == s: 
                # Assigne 1 si le mot actuel est dans la position de vocabulaire
                bag[i] = 1
    return(np.array(bag))

def predictClass(sentence, model):
    # Filtrer les prédictions
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Trier par force de probabilité
    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({"intent": classes[r[0]], "probability": str(r[1])})
    return returnList


# getresponse fonction pour obtenir la réponse de l'intention
def getResponse(ints, intentsJson):
    tag = ints[0]['intent']
    list_of_intents = intentsJson['intents']
    
    for i in listOfIntents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# fonction pour les reponse du bot
def chatbotResponse(msg):
    ints = predictClass(msg, model)
    res = getResponse(ints, intents)
    return res


def mainChat():
    print("Parle")
    while True:
        message = input("")
        if message == "quit":
            break
        print(chatbotResponse(message))
        
        
mainChat()