import sys
from PySide6 import QtWidgets, QtCore
from chat import MainWidget

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyleSheet("QWidget { background-color: #344D59; }")
    QtCore.QCoreApplication.setApplicationName("")
    widget = MainWidget()
    widget.show()
    widget.resize(800, 600)
    
    sys.exit(app.exec())
