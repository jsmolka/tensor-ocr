from PyQt5.QtCore import QByteArray, QPoint, QSettings, QSize
from PyQt5.QtGui import QCloseEvent, QResizeEvent
from PyQt5.QtWidgets import QWidget


class MainWindow(QWidget):
    def __init__(self, parent=None):
        """Constructor."""
        QWidget.__init__(self, parent=parent)
        self._size = QSize()

    def _save_state(self):
        """Saves the current window state."""
        settings = QSettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("pos", self.pos())
        settings.setValue("size", self._size if self.isMaximized() else self.size())
        settings.setValue("maximized", self.isMaximized())

    def show_saved_state(self):
        """Loads the previously saved window state and shows the widget."""
        settings = QSettings()
        self.restoreGeometry(settings.value("geometry", self.saveGeometry(), type=QByteArray))
        self.move(settings.value("pos", self.pos(), type=QPoint))
        self._size = settings.value("size", self.size(), type=QSize)
        self.resize(self._size)
        if settings.value("maximized", self.isMaximized(), type=bool):
            self.showMaximized()
        else:
            self.show()

    def closeEvent(self, event):
        """Overriden close event."""
        self._save_state()
        QWidget.closeEvent(self, event)

    def resizeEvent(self, event):
        """Overriden resize event."""
        self._size = event.oldSize()
        QWidget.resizeEvent(self, event)
