from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QLabel


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        """Constructor."""
        QLabel.__init__(self, parent=parent)
        self._pixmap = QPixmap()
        self._path = ""

    def load(self, path):
        """Loads the pixmap from a given path."""
        self._path = path
        self._pixmap = QPixmap(path)
        self._update()
        
    def path(self):
        """Path of the current pixmap."""
        return self._path

    def _scaled_pixmap(self):
        """Resizes the pixmap inside the label."""
        pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Center the pixmap horizontally. Centering it vertically is done
        # automatically
        if pixmap.width() != self.width():
            fixed = QPixmap(self.size())
            fixed.fill(Qt.transparent)
            
            painter = QPainter(fixed)
            painter.drawPixmap((self.width() - pixmap.width()) // 2, 0, pixmap)
            return fixed

        return pixmap

    def _update(self):
        """Updates the labels pixmap."""
        if not self._pixmap.isNull():
            self.setPixmap(self._scaled_pixmap())

    def resizeEvent(self, event):
        """Overriden resize event."""
        QLabel.resizeEvent(self, event)
        self._update()

