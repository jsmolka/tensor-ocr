import resources
import sys
from PyQt5.QtWidgets import QApplication
from traceback import format_exception
from widgets.tensorwidget import TensorWidget


def except_hook(*exc):
    """Global except hook."""
    print("".join(format_exception(*exc)))
    

def main():
    """Main function."""
    sys.excepthook = except_hook

    app = QApplication(sys.argv)
    app.setApplicationName("tensor-ocr")
    app.setOrganizationName("tensor-ocr inc.")
    
    widget = TensorWidget()
    widget.show_saved_state()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
