from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QLabel


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        """Constructor."""
        QLabel.__init__(self, parent=parent)
        self.pixmap = QPixmap()
        self.path = ""

    def load(self, path):
        """Loads the pixmap from a given path."""
        self.path = path
        self.pixmap = QPixmap(path)
        self.update()
        
    def scaled_pixmap(self):
        """Resizes the pixmap inside the label."""
        pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if pixmap.width() != self.width():
            fixed = QPixmap(self.size())
            fixed.fill(Qt.transparent)
            
            painter = QPainter(fixed)
            painter.drawPixmap((self.width() - pixmap.width()) // 2, 0, pixmap)
            return fixed

        return pixmap

    def update(self):
        """Updates the labels pixmap."""
        if not self.pixmap.isNull():
            self.setPixmap(self.scaled_pixmap())

    def resizeEvent(self, event):
        """Overriden resize event."""
        QLabel.resizeEvent(self, event)
        self.update()

