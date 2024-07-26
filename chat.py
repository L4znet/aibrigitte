import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import sys
from PySide6 import QtWidgets
try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    print("PySide6 is not installed")
    sys.exit(1)


from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.layoutMain = QtWidgets.QVBoxLayout() 
        
        self.title = QtWidgets.QLabel("Brigitte")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        self.inputQuestion = QtWidgets.QLineEdit()
        self.inputQuestion.setPlaceholderText("Ecrit un truc")
        
        self.buttonSend = QtWidgets.QPushButton("Envoyer")
        
        self.chatBox = QtWidgets.QVBoxLayout()
        self.chatBoxWidget = QtWidgets.QWidget()
        self.chatBoxWidget.setLayout(self.chatBox)
        
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.chatBoxWidget)


        self.layoutMain.addLayout(self.chatBox)
        self.layoutMain.addWidget(self.title)
        self.layoutMain.addWidget(self.inputQuestion)
        self.layoutMain.addWidget(self.buttonSend)
        self.layoutMain.addWidget(self.scrollArea)
        
        self.setLayout(self.layoutMain)
        
        
        self.buttonSend.clicked.connect(self.sendUserMessage)
        self.inputQuestion.returnPressed.connect(self.sendUserMessage)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(10, 26, 42, 0.8);
                color: #e67e22;
                font-family: Arial, sans-serif;
                font-size: 16px;
            }
            QLineEdit {
                background-color: rgba(10, 26, 42, 0.8);
                color: #e67e22;
                border: 1px solid #e67e22;
                padding: 5px;
            }
            QPushButton {
                background-color: #e67e22;
                color: #0a1a2a;
                border: none;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #e67e22;
            }
            QScrollArea {
                background-color: rgba(10, 26, 42, 0.8);
            }
        """)

        
    @QtCore.Slot()
    
    

    def cleanUpSentence(self,sentence):
        # Tokenise la phrase - séparer les mots dans un tableau
        sentenceWords = nltk.word_tokenize(sentence)
        # Lemmatizer chaque mot
        sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
        return sentenceWords

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, words):
        # Tokeniser la phrase
        sentenceWords = self.cleanUpSentence(sentence)
        # Sac de mots, matrice de vocabulaire
        bag = [0]*len(words)  
        for s in sentenceWords:
            for i,w in enumerate(words):
                if w == s: 
                    # Assigne 1 si le mot actuel est dans la position de vocabulaire
                    bag[i] = 1
        return(np.array(bag))

    def predictClass(self,sentence, model):
        # Filtrer les prédictions
        p = self.bow(sentence, words)
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
    def getResponse(self,ints, intentsJson):
        tag = ints[0]['intent']
        listOfIntents = intentsJson['intents']
        
        for i in listOfIntents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    # fonction pour les reponse du bot
    def chatbotResponse(self,msg):
        ints = self.predictClass(msg, model)
        res = self.getResponse(ints, intents)
        return res


    def sendUserMessage(self):
            user_message = self.inputQuestion.text()
            if user_message:
                self.chatBox.addWidget(QtWidgets.QLabel(f"User: {user_message}"))
                response_message = self.chatbotResponse(user_message)
                self.chatBox.addWidget(QtWidgets.QLabel(f"Bot: {response_message}"))
                self.inputQuestion.clear()
                self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
        

    # def mainChat(self):
    #     print("Parle")
    #     while True:
    #         message = self.inputQuestion.text()
    #         if message == "quit":
    #             break
    #         print(self.chatbotResponse(message))
            #  self.chatBox.addWidget(QtWidgets.QLabel(self.chatbotResponse(message)))