from os.path import dirname, exists
from PyQt5.QtCore import QFile, QTextStream, QSize, Qt, QTextStream
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from model.interface import init_model, predict_word
from widgets.imagelabel import ImageLabel
from widgets.mainwindow import MainWindow


class TensorWidget(MainWindow):
    def __init__(self, parent=None):
        """Constructor."""
        MainWindow.__init__(self, parent=parent)
        self.image = ImageLabel(parent=self)
        self.load = QPushButton(parent=self)
        self.convert = QPushButton(parent=self)
        self.save = QPushButton(parent=self)
        self.text = QLabel(parent=self)

        self.setup()
        self.setup_ui()

        self.load.released.connect(self.load_released)
        self.convert.released.connect(self.convert_released)
        self.save.released.connect(self.save_released)

        init_model()

    def setup(self):
        """Sets up the widget."""
        qfile = QFile(":/resource/tensor.css")
        if qfile.open(QFile.ReadOnly):
            stream = QTextStream(qfile)
            self.setStyleSheet(stream.readAll())

    def setup_ui(self):
        """Sets up user interface."""
        self.load.setText("Load")
        self.convert.setText("Convert")
        self.save.setText("Save")

        self.image.setMinimumSize(QSize(130, 130))
        self.load.setMinimumSize(QSize(75, 33))
        self.convert.setMinimumSize(QSize(75, 33))
        self.save.setMinimumSize(QSize(75, 33))
        self.text.setMinimumSize(QSize(130, 130))

        policy = self.image.sizePolicy()
        policy.setHorizontalStretch(1)
        self.image.setSizePolicy(policy)
        policy = self.text.sizePolicy()
        policy.setHorizontalStretch(1)
        self.text.setSizePolicy(policy)

        self.text.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.load)
        vbox.setAlignment(self.load, Qt.AlignBottom)
        vbox.addWidget(self.convert)
        vbox.addWidget(self.save)
        vbox.setAlignment(self.save, Qt.AlignTop)

        hbox = QHBoxLayout()
        hbox.setSpacing(15)
        hbox.addWidget(self.image)
        hbox.addLayout(vbox)
        hbox.addWidget(self.text)
        self.setLayout(hbox)

    def load_released(self):
        """Action performed on load released."""
        dialog = QFileDialog(parent=self)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dialog.exec_():
            self.image.load(dialog.selectedFiles()[0])

    def convert_released(self):
        """Action performed on convert released."""
        if exists(self.image.path):
            self.text.setText(predict_word(self.image.path))

    def save_text(self, path):
        """Saves the text to a given path."""
        qfile = QFile(path)
        if qfile.open(QFile.WriteOnly):
            stream = QTextStream(qfile)
            stream << self.text.text()

    def save_released(self):
        """Action performned on save released."""
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Text (*.txt)")
        dialog.selectFile("text.txt")
        if dialog.exec_():
            self.save_text(dialog.selectedFiles()[0])
