import sys
from PySide6 import QtWidgets, QtCore
from chat import MainWidget


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyleSheet("QWidget  { background-color: #0C1821; }")
    QtCore.QCoreApplication.setApplicationName("Brigitte")
    widget= MainWidget()
    widget.show()
    widget.resize(800, 600)
    
    sys.exit(app.exec())