import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout
, QPushButton, QSizePolicy, QLabel, QFontDialog)


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        btn = QPushButton('Dialog', self)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.move(20, 20)
        btn.clicked.connect(self.showDialog)

        vbox = QVBoxLayout()
        vbox.addWidget(btn)

        self.lbl = QLabel('Mother Fucker', self)
        self.lbl.move(130, 20)

        vbox.addWidget(self.lbl)
        self.setLayout(vbox)

        self.setWindowTitle('Font Dialog')
        self.setGeometry(300, 300, 250, 180)
        self.show()

    def showDialog(self):
        font, ok = QFontDialog.getFont()

        if ok:
           self.lbl.setFont(font)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())