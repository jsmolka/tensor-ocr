import datetime as dt
import sys
import traceback as tb
from PyQt5.QtWidgets import QApplication
from widgets.mainwindow import MainWindow


def except_hook(*exc):
    """Global except hook."""
    msg = "".join(tb.format_exception(*exc))
    print(msg)
    log(msg)


def log(msg):
    """Logs a message."""
    ts = dt.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open("tensor.log", "a+") as logfile:
        logfile.write("{}: {}\n".format(ts, msg))


def main():
    """Main function."""
    sys.excepthook = except_hook

    app = QApplication(sys.argv)
    app.setApplicationName("tensor-ocr")
    app.setOrganizationName("tensor-ocr inc.")

    window = MainWindow()
    window.show_saved_state()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
