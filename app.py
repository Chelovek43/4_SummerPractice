import os
import sys
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox

from ui.ui import FootballPredictorApp

# Динамически определяемый путь для журнала логов
log_path = os.path.join(os.path.expanduser("~"), "football_predictor_error.log")
logging.basicConfig(filename=log_path, level=logging.ERROR)

def exception_hook(exctype, value, traceback):
    # Журнал логов для ошибок и непредвиденных исключений
    logging.error("Необработанное исключение", exc_info=(exctype, value, traceback))

    # Делаю фокус сверху (главное окно)
    parent = QApplication.activeWindow() if QApplication.instance() else None
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Непредвиденная ошибка!")
    msg.setText(f"Произошла непредвиденная ошибка:\n{value}")
    msg.setDetailedText(''.join(traceback.format_exception(exctype, value, traceback)))
    msg.exec()
    sys.__excepthook__(exctype, value, traceback)

if __name__ == "__main__":
    
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    
    # Установка стиля  интерфейса
    app.setStyle("Fusion")
    
    window = FootballPredictorApp()
    window.show()
    sys.exit(app.exec())