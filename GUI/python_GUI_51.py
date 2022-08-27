
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Reimplementing event handler')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_F:
            self.showFullScreen()
        elif e.key() == Qt.Key_N:
            self.showNormal()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


'''
이벤트 핸들러	설명
keyPressEvent	키보드를 눌렀을 때 동작합니다.
keyReleaseEvent	키보드를 눌렀다가 뗄 때 동작합니다.
mouseDoubleClickEvent	마우스를 더블클릭할 때 동작합니다.
mouseMoveEvent	마우스를 움직일 때 동작합니다.
mousePressEvent	마우스를 누를 때 동작합니다.
mouseReleaseEvent	마우스를 눌렀다가 뗄 때 동작합니다.
moveEvent	위젯이 이동할 때 동작합니다.
resizeEvent	위젯의 크기를 변경할 때 동작합니다.
'''