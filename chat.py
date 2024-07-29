import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import sys
from tensorflow.keras.models import load_model
from PySide6 import QtWidgets
import json
import random

try:
    from PySide6 import QtCore, QtWidgets, QtGui
except ImportError:
    print("PySide6 is not installed")
    sys.exit(1)

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
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
        self.layoutMain.addWidget(self.inputQuestion)
        self.layoutMain.addWidget(self.buttonSend)
        self.layoutMain.addWidget(self.scrollArea)
        
        self.setLayout(self.layoutMain)
        
        self.buttonSend.clicked.connect(self.sendUserMessage)
        self.inputQuestion.returnPressed.connect(self.sendUserMessage)
        
        self.setStyleSheet(f"""
            MainWidget {{
                background: url('./evennement.jpg') no-repeat center center fixed;
                background-size: cover;
            }}
            QWidget {{
                background-color: #344D59; /* Couleur principale pour le fond */
                color: #B8CBD0; /* Couleur du texte principale */
                font-family: Arial, sans-serif;
                font-size: 16px;
            }}
            QLineEdit {{
                background-color: #709CA7; /* Couleur d'arrière-plan pour les champs de texte */
                color: #344D59; /* Couleur du texte dans les champs de texte */
                border: 1px solid #7A90A4; /* Couleur de la bordure */
                padding: 5px;
            }}
            QPushButton {{
                background-color: #137C8B; /* Couleur d'arrière-plan pour les boutons */
                color: #B8CBD0; /* Couleur du texte des boutons */
                border: none;
                padding: 10px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #7A90A4; /* Couleur d'arrière-plan au survol */
            }}
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                color: #B8CBD0; /* Couleur du texte des labels */
            }}
            QScrollArea {{
                background-color: transparent; /* Background of the scroll area is transparent */
                border: none;
            }}
            QScrollBar:vertical {{
                border: none;
                background: qlineargradient(
                    x1: 0, y1: 0, 
                    x2: 0, y2: 1, 
                    stop: 0 #709CA7, 
                    stop: 1 #137C8B
                );
                width: 14px;
                margin: 15px 3px 15px 3px;
                border-radius: 7px;
            }}
            QScrollBar::handle:vertical {{
                background: #B8CBD0; /* Couleur d'arrière-plan du curseur */
                border: 2px solid #7A90A4; /* Bordure du curseur */
                border-radius: 7px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:disabled {{
                background: rgba(30, 60, 114, 0.5); /* Couleur d'arrière-plan du curseur désactivé */
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
                border: none;
                width: 0px;
                height: 0px;
                background: none;
            }}
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
            self.smoothScrollToBottom()

    def smoothScrollToBottom(self):
        scrollBar = self.scrollArea.verticalScrollBar()
        animation = QtCore.QPropertyAnimation(scrollBar, b"value")
        animation.setDuration(500)  # Duration of the animation in milliseconds
        animation.setStartValue(scrollBar.value())
        animation.setEndValue(scrollBar.maximum())
        animation.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        animation.start()
        # Keep a reference to the animation object to prevent it from being garbage collected
        self.scrollAnimation = animation
        

    # def mainChat(self):
    #     print("Parle")
    #     while True:
    #         message = self.inputQuestion.text()
    #         if message == "quit":
    #             break
    #         print(self.chatbotResponse(message))
            #  self.chatBox.addWidget(QtWidgets.QLabel(self.chatbotResponse(message)))