from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout
from widgets.mainwindow import MainWindow


class TensorWidget(MainWindow):
    def __init__(self, parent=None):
        """Constructor."""
        MainWindow.__init__(self, parent=parent)
        self._label = QLabel(parent=self)
        self._load = QPushButton(parent=self)
        self._convert = QPushButton(parent=self)
        self._text = QTextEdit(parent=self)

        self._setup()
        self._setup_ui()

        self._load.clicked.connect(self._load_clicked)
        self._convert.clicked.connect(self._convert_clicked)

    def _setup(self):
        """Sets up widget."""
        self.setWindowTitle("Tensor OCR")

    def _setup_ui(self):
        """Sets up user interface."""
        self._load.setText("Load")
        self._convert.setText("Convert")
        self._text.setReadOnly(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self._load)
        vbox.addWidget(self._convert)

        hbox = QHBoxLayout()
        hbox.addWidget(self._label)
        hbox.addLayout(vbox)
        hbox.addWidget(self._text)
        self.setLayout(hbox)

    def _load_clicked(self):
        """Action performed on load clicked."""
        pass

    def _convert_clicked(self):
        """Action performed on convert clicked."""
        pass
