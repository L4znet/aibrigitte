import sys
import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from PySide6 import QtWidgets, QtCore, QtGui
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layoutMain = QtWidgets.QVBoxLayout()

        # Ajouter l'image du chatbot au centre
        self.chatbot_image = QtWidgets.QLabel()
        self.set_chatbot_image('chatbot.jpg')  
        self.chatbot_image.setAlignment(QtCore.Qt.AlignCenter)
        self.layoutMain.addWidget(self.chatbot_image, alignment=QtCore.Qt.AlignCenter)

        # Mise en page de la boîte de chat
        self.chatBox = QtWidgets.QVBoxLayout()
        self.chatBox.setAlignment(QtCore.Qt.AlignTop)
        self.chatBoxWidget = QtWidgets.QWidget()
        self.chatBoxWidget.setLayout(self.chatBox)
        self.chatBoxWidget.setStyleSheet("background-color: white; border-radius: 10px;")
        self.chatBoxScroll = QtWidgets.QScrollArea()
        self.chatBoxScroll.setWidget(self.chatBoxWidget)
        self.chatBoxScroll.setWidgetResizable(True)
        self.chatBoxScroll.setStyleSheet("""
            border: none;
            QScrollBar:vertical {
                border: none;
                background-color: #E1E1E1;
                width: 14px;
                margin: 0px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background-color: #D96F4C;
                min-height: 30px;
                border-radius: 7px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                subcontrol-origin: margin;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        self.layoutMain.addWidget(self.chatBoxScroll)

        # Champ de saisie et bouton d'envoi
        self.inputQuestion = QtWidgets.QLineEdit()
        self.inputQuestion.setPlaceholderText("Écris quelque chose...")
        self.inputQuestion.setStyleSheet("""
            background-color: #B8CBD0;
            color: black;
            border: 1px solid #CCCCCC;
            border-radius: 15px;
            padding: 10px;
            margin-right: 10px;
        """)

        # Bouton avec icône d'envoi
        self.buttonSend = QtWidgets.QPushButton()
        send_icon = QtGui.QIcon("send_icon.png") 
        self.buttonSend.setIcon(send_icon)
        self.buttonSend.setStyleSheet("""
            background-color: #1F272B;
            border-radius: 15px;
            padding: 10px;
        """)

        # Mise en page pour le champ de saisie et le bouton
        self.inputLayout = QtWidgets.QHBoxLayout()
        self.inputLayout.addWidget(self.inputQuestion)
        self.inputLayout.addWidget(self.buttonSend)
        self.layoutMain.addLayout(self.inputLayout)

        self.setLayout(self.layoutMain)

        # Connecter le clic du bouton à la méthode sendUserMessage
        self.buttonSend.clicked.connect(self.sendUserMessage)
        self.inputQuestion.returnPressed.connect(self.sendUserMessage) 

        # Définir le titre de la fenêtre et la taille
        self.setWindowTitle("Chatbot Brigitte")
        self.resize(800, 600)

    def set_chatbot_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        # Redimensionner et recadrer pour s'adapter dans un cercle
        size = min(pixmap.width(), pixmap.height())
        rect = QtCore.QRect((pixmap.width() - size) // 2, (pixmap.height() - size) // 2, size, size)
        cropped_pixmap = pixmap.copy(rect).scaled(100, 100, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)

        # Créer un masque circulaire
        mask = QtGui.QPixmap(100, 100)
        mask.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(mask)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtCore.Qt.black)
        painter.drawEllipse(0, 0, 100, 100)
        painter.end()

        # Appliquer le masque à l'image
        cropped_pixmap.setMask(mask.createMaskFromColor(QtCore.Qt.transparent))

        # Définir le pixmap sur le label
        self.chatbot_image.setPixmap(cropped_pixmap)
        self.chatbot_image.setFixedSize(100, 100)

    @QtCore.Slot()
    def cleanUpSentence(self, sentence):
        # Tokeniser la phrase - diviser les mots en un tableau
        sentenceWords = nltk.word_tokenize(sentence)
        # Lemmatizer chaque mot
        sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
        return sentenceWords

    def bow(self, sentence, words):
        # Tokeniser la phrase
        sentenceWords = self.cleanUpSentence(sentence)
        # Sac de mots, matrice de vocabulaire
        bag = [0] * len(words)  
        for s in sentenceWords:
            for i, w in enumerate(words):
                if w == s: 
                    bag[i] = 1
        return np.array(bag)

    def predictClass(self, sentence, model):
        # Filtrer les prédictions
        p = self.bow(sentence, words)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # Trier par force de probabilité
        results.sort(key=lambda x: x[1], reverse=True)
        returnList = []
        for r in results:
            returnList.append({"intent": classes[r[0]], "probability": str(r[1])})
        return returnList

    def getResponse(self, ints, intentsJson):
        tag = ints[0]['intent']
        listOfIntents = intentsJson['intents']
        
        for i in listOfIntents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbotResponse(self, msg):
        ints = self.predictClass(msg, model)
        res = self.getResponse(ints, intents)
        return res

    def sendUserMessage(self):
        user_message = self.inputQuestion.text()
        if user_message:
            # Ajouter le message de l'utilisateur à la boîte de chat
            self.chatBox.addWidget(self.create_message_bubble(user_message, is_user=True))
            reponseIA = self.chatbotResponse(user_message)
            self.chatBox.addWidget(self.create_message_bubble(reponseIA, is_user=False))
            self.inputQuestion.clear()
            self.chatBoxScroll.verticalScrollBar().setValue(self.chatBoxScroll.verticalScrollBar().maximum())

    def create_message_bubble(self, message, is_user=True):
        # Créer un QWidget pour la bulle de message
        bubble = QtWidgets.QWidget()
        bubble_layout = QtWidgets.QHBoxLayout(bubble)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        bubble_layout.setAlignment(QtCore.Qt.AlignLeft if not is_user else QtCore.Qt.AlignRight)

        # Créer un QLabel pour afficher le message
        message_label = QtWidgets.QLabel(message)
        message_label.setWordWrap(True)
        message_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        message_label.setStyleSheet("""
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
        """)

        if is_user:
            message_label.setStyleSheet("""
                background-color: #709CA7;
                color: black;
                font-size: 15px;
                border-radius: 15px;
                padding: 7px;
                margin: 4px;
                max-width: 500%; /* Ajustez si nécessaire */
            """)
            bubble_layout.addWidget(message_label)
            # bubble.setStyleSheet("align-self: flex-end;")
        else:
            message_label.setStyleSheet("""
                background-color: #7A90A4;
                color: black;
                font-size: 15px;
                border-radius: 15px;
                padding: 7px;
                margin: 4px;
                max-width: 500%; /* Ajustez si nécessaire */
            """)
            bubble_layout.addWidget(message_label)
            # bubble.setStyleSheet("align-self: flex-start;")

        return bubble

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyleSheet("QWidget { background-color: #E1E1E1; }")
    QtCore.QCoreApplication.setApplicationName("Brigitte")
    widget = MainWidget()
    widget.show()
    
    sys.exit(app.exec())
