from os.path import dirname
from PyQt5.QtCore import QFile, QTextStream, QSize, Qt, QTextStream
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout

from model.model_interface import init_model, probable_words
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
        self.text = QTextEdit(parent=self)

        self.setup()
        self.setup_ui()

        self.load.clicked.connect(self.load_clicked)
        self.convert.clicked.connect(self.convert_clicked)
        self.save.clicked.connect(self.save_clicked)

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
        self.text.setReadOnly(True)

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

    def load_clicked(self):
        """Action performed on load clicked."""
        dialog = QFileDialog(parent=self)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dialog.exec_():
            self.image.load(dialog.selectedFiles()[0])

    def convert_clicked(self):
        """Action performed on convert clicked."""
        if not self.image.path:
            return

        self.text.clear()
        words = probable_words(self.image.path)
        for word in words:
            self.text.append(word)

    def save_text(self, path):
        """Saves the text to a given path."""
        qfile = QFile(path)
        if qfile.open(QFile.WriteOnly):
            stream = QTextStream(qfile)
            stream << self.text.toPlainText()

    def save_clicked(self):
        """Action performned on save clicked."""
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Text (*.txt)")
        dialog.selectFile("text.txt")
        if dialog.exec_():
            self.save_text(dialog.selectedFiles()[0])
