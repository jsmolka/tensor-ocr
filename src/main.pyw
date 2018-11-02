import resources
import sys
from PyQt5.QtWidgets import QApplication
from traceback import format_exception
from widgets.tensorwidget import TensorWidget


def excepthook(*exc):
    """Global except hook."""
    print("".join(format_exception(*exc)))
    

def main():
    """Main function."""
    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    # Needed for QSettings to work
    app.setApplicationName("tensor-ocr")
    app.setOrganizationName("tensor-ocr inc.")
    
    tensor = TensorWidget()
    tensor.show_saved_state()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
