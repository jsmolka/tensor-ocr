from os.path import exists
from PyQt5.QtCore import QFile, QSettings, QSize, Qt, QTextStream
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from model.interface import init_model, predict_word
from widgets.imagelabel import ImageLabel
from widgets.mainwindow import MainWindow


class TensorWidget(MainWindow):
    def __init__(self, parent=None):
        """Constructor."""
        MainWindow.__init__(self, parent=parent)
        self.image_lbl = ImageLabel(parent=self)
        self.load_btn = QPushButton(parent=self)
        self.convert_btn = QPushButton(parent=self)
        self.save_btn = QPushButton(parent=self)
        self.text_lbl = QLabel(parent=self)
        self.dict_chk = QCheckBox(parent=self)

        self.last_dir = ""

        self.setup()
        self.setup_ui()
        self.load_state()

        self.load_btn.released.connect(self.load_released)
        self.convert_btn.released.connect(self.convert_released)
        self.save_btn.released.connect(self.save_released)

        init_model()

    def setup(self):
        """Sets up the widget."""
        qfile = QFile(":/resource/tensor.css")
        if qfile.open(QFile.ReadOnly):
            stream = QTextStream(qfile)
            self.setStyleSheet(stream.readAll())

    def setup_ui(self):
        """Sets up user interface."""
        self.load_btn.setText("Load")
        self.convert_btn.setText("Convert")
        self.save_btn.setText("Save")
        self.dict_chk.setText("Use dictionary")

        size = QSize(175, 175)
        self.image_lbl.setMinimumSize(size)
        self.text_lbl.setMinimumSize(size)

        size = QSize(135, 33)
        self.load_btn.setMinimumSize(size)
        self.convert_btn.setMinimumSize(size)
        self.save_btn.setMinimumSize(size)
        self.dict_chk.setMinimumSize(size)

        policy = self.image_lbl.sizePolicy()
        policy.setHorizontalStretch(1)
        self.image_lbl.setSizePolicy(policy)
        policy = self.text_lbl.sizePolicy()
        policy.setHorizontalStretch(1)
        self.text_lbl.setSizePolicy(policy)

        self.text_lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.load_btn)
        vbox.setAlignment(self.load_btn, Qt.AlignBottom)
        vbox.addWidget(self.convert_btn)
        vbox.addWidget(self.save_btn)
        vbox.addWidget(self.dict_chk)
        vbox.setAlignment(self.dict_chk, Qt.AlignTop)

        hbox = QHBoxLayout()
        hbox.setSpacing(15)
        hbox.addWidget(self.image_lbl)
        hbox.addLayout(vbox)
        hbox.addWidget(self.text_lbl)
        self.setLayout(hbox)

    def load_released(self):
        """Action performed on load button released."""
        dialog = QFileDialog(parent=self)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if self.last_dir:
            dialog.setDirectory(self.last_dir)
        if dialog.exec_():
            self.image_lbl.load(dialog.selectedFiles()[0])
            self.last_dir = dialog.directory().absolutePath()

    def convert_released(self):
        """Action performed on convert button released."""
        if exists(self.image_lbl.path):
            self.text_lbl.setText(predict_word(self.image_lbl.path, self.dict_chk.isChecked()))

    def save_text(self, path):
        """Saves the text label content to a given path."""
        qfile = QFile(path)
        if qfile.open(QFile.WriteOnly):
            stream = QTextStream(qfile)
            stream << self.text_lbl.text()

    def save_released(self):
        """Action performned on save button released."""
        dialog = QFileDialog(parent=self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Text (*.txt)")
        dialog.selectFile("word.txt")
        if dialog.exec_():
            self.save_text(dialog.selectedFiles()[0])

    def load_state(self):
        """Loads the user input of the previous session."""
        settings = QSettings()
        self.last_dir = settings.value("lastDir", None, type=str)
        self.dict_chk.setChecked(settings.value("useDict", True, type=bool))

    def save_state(self):
        """Saves the user input for the next session."""
        settings = QSettings()
        if exists(self.last_dir):
            settings.setValue("lastDir", self.last_dir)
        settings.setValue("useDict", self.dict_chk.isChecked())

    def closeEvent(self, event):
        """Overriden close event."""
        self.save_state()
        MainWindow.closeEvent(self, event)
