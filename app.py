import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt
import pandas as pd

from core.predicrtor import FootballMatchPredictor

class FootballPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = FootballMatchPredictor("football.csv")
        self.init_ui()
        self.setWindowTitle("Football Match Predictor")
        self.setMinimumSize(600, 500)
        
    def init_ui(self):
        # Главный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Заголовок
        title = QLabel("Football Match Predictor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        main_layout.addWidget(title)
        
        # Выбор команд
        team_selection_layout = QHBoxLayout()
        
        # Домашняя команда
        home_layout = QVBoxLayout()
        home_layout.addWidget(QLabel("Домашняя команда:"))
        self.home_combo = QComboBox()
        home_layout.addWidget(self.home_combo)
        
        # Гостевая команда
        away_layout = QVBoxLayout()
        away_layout.addWidget(QLabel("Гостевая команда:"))
        self.away_combo = QComboBox()
        away_layout.addWidget(self.away_combo)
        
        team_selection_layout.addLayout(home_layout)
        team_selection_layout.addLayout(away_layout)
        main_layout.addLayout(team_selection_layout)
        
        # Кнопка прогноза
        self.predict_btn = QPushButton("Прогноз")
        self.predict_btn.clicked.connect(self.predict_match)
        self.predict_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-weight: bold; background-color: #4CAF50; color: white; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        main_layout.addWidget(self.predict_btn)
        
        # Область вывода результатов
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-family: monospace;")
        main_layout.addWidget(self.result_display)
        
        # Заполняем списки команд
        self.load_teams()
        
    def load_teams(self):
        #Загружает списки команд в выпадающие меню

        home_teams, away_teams = self.predictor.get_team_list()
        
        self.home_combo.clear()
        self.away_combo.clear()
        
        self.home_combo.addItems(home_teams)
        self.away_combo.addItems(away_teams)
        
        # Устанавливаем первые команды по умолчанию
        if home_teams:
            self.home_combo.setCurrentIndex(0)
        if away_teams:
            self.away_combo.setCurrentIndex(0 if len(away_teams) == 1 else 1)
    
    def predict_match(self):
        #Обработчик нажатия кнопки прогноза

        home_team = self.home_combo.currentText()
        away_team = self.away_combo.currentText()
        
        if home_team == away_team:
            QMessageBox.warning(self, "Ошибка", "Команды не должны быть одинаковыми!")
            return
        
        try:
            prediction = self.predictor.predict_match(home_team, away_team)
            self.display_prediction(prediction)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка прогноза: {str(e)}")
    
    def display_prediction(self, prediction):
        #Отображает результаты прогноза

        home = prediction['teams']['home']
        away = prediction['teams']['away']
        home_goals = prediction['expected_score']['home']
        away_goals = prediction['expected_score']['away']
        
        text = f"=== Match Prediction: {home} vs {away} ===\n\n"
        text += f"Expected score: {home_goals} - {away_goals}\n\n"
        text += "Outcome probabilities:\n"
        text += f"- {home} win: {prediction['outcome_probabilities']['home_win']*100:.1f}%\n"
        text += f"- Draw: {prediction['outcome_probabilities']['draw']*100:.1f}%\n"
        text += f"- {away} win: {prediction['outcome_probabilities']['away_win']*100:.1f}%\n\n"
        text += "Most probable scores:\n"
        
        for score, prob in prediction['likely_scores'].items():
            text += f"{score[0]}-{score[1]}: {prob*100:.2f}%\n"
        
        self.result_display.setPlainText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Установка стиля  интерфейса
    app.setStyle("Fusion")
    
    window = FootballPredictorApp()
    window.show()
    sys.exit(app.exec())