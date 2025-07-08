import sys
from PyQt6.QtWidgets import QApplication

from ui.ui import FootballPredictorApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Установка стиля  интерфейса
    app.setStyle("Fusion")
    
    window = FootballPredictorApp()
    window.show()
    sys.exit(app.exec())

