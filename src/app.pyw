import sys
import traceback as tb
from PyQt5.QtWidgets import QApplication

import data.resources
from widgets.tensorwidget import TensorWidget


def main(argv):
    """Main function."""
    sys.excepthook = lambda *x: print("".join(tb.format_exception(*x)))

    app = QApplication(argv)
    app.setApplicationName("tensor-ocr")
    app.setOrganizationName("tensor-ocr inc.")
    
    tensor = TensorWidget()
    tensor.show_saved_state()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
