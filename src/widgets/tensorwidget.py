from PyQt5.QtCore import QFile, QTextStream, QSize, Qt, QTextStream
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout
from widgets.imagelabel import ImageLabel
from widgets.mainwindow import MainWindow


class TensorWidget(MainWindow):
    def __init__(self, parent=None):
        """Constructor."""
        MainWindow.__init__(self, parent=parent)
        self._image = ImageLabel(parent=self)
        self._load = QPushButton(parent=self)
        self._convert = QPushButton(parent=self)
        self._save = QPushButton(parent=self)
        self._text = QTextEdit(parent=self)

        self._setup()
        self._setup_ui()

        self._load.clicked.connect(self._load_clicked)
        self._convert.clicked.connect(self._convert_clicked)
        self._save.clicked.connect(self._save_clicked)

    def _setup(self):
        """Sets up the widget."""
        qfile = QFile(":/resource/tensor.css")
        if qfile.open(QFile.ReadOnly):
            stream = QTextStream(qfile)
            self.setStyleSheet(stream.readAll())

    def _setup_ui(self):
        """Sets up user interface."""
        self._load.setText("Load")
        self._convert.setText("Convert")
        self._save.setText("Save")
        self._text.setReadOnly(True)

        self._image.setMinimumSize(QSize(130, 130))
        self._load.setMinimumSize(QSize(75, 33))
        self._convert.setMinimumSize(QSize(75, 33))
        self._save.setMinimumSize(QSize(75, 33))
        self._text.setMinimumSize(QSize(130, 130))

        # Make sure that the image and the textbox are the same size
        policy = self._image.sizePolicy()
        policy.setHorizontalStretch(1)
        self._image.setSizePolicy(policy)
        policy = self._text.sizePolicy()
        policy.setHorizontalStretch(1)
        self._text.setSizePolicy(policy)

        vbox = QVBoxLayout()
        vbox.addWidget(self._load)
        vbox.setAlignment(self._load, Qt.AlignBottom)
        vbox.addWidget(self._convert)
        vbox.addWidget(self._save)
        vbox.setAlignment(self._save, Qt.AlignTop)

        hbox = QHBoxLayout()
        hbox.setSpacing(15)
        hbox.addWidget(self._image)
        hbox.addLayout(vbox)
        hbox.addWidget(self._text)
        self.setLayout(hbox)

    def _load_clicked(self):
        """Action performed on load clicked."""
        dialog = QFileDialog(parent=self)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dialog.exec_():
            self._image.load(dialog.selectedFiles()[0])

    def _convert_clicked(self):
        """Action performed on convert clicked."""
        pass

    def _save_text(self, path):
        """Saves the text to a given path."""
        qfile = QFile(path)
        if qfile.open(QFile.WriteOnly):
            stream = QTextStream(qfile)
            stream << self._text.toPlainText()

    def _save_clicked(self):
        """Action performned on save clicked."""
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Text (*.txt)")
        dialog.selectFile("text.txt")
        if dialog.exec_():
            self._save_text(dialog.selectedFiles()[0])