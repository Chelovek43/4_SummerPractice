import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt
import pandas as pd

from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtGui import QDoubleValidator

from core.predicrtor import FootballMatchPredictor
from core.odds_predicror import OddsMatchPredictor

class FootballPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = FootballMatchPredictor("football.csv")

        self.predictor_odds = OddsMatchPredictor("Laliga.csv")
        self.predictor_odds.train_model()  # Обучаем модель при инициализации

        self.init_ui()
        self.setWindowTitle("Football Match Predictor")
        self.setMinimumSize(600, 500)
        
    def init_ui(self):
        # Главный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        self.main_layout = QVBoxLayout(central_widget)
        
        # Заголовок
        title = QLabel("Football Match Predictor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        self.main_layout.addWidget(title)
        
        # Переключатель режимов 
        self.setup_mode_switch()
        
        # Поля для коэффициентов (изначально скрыты)
        self.setup_odds_inputs()
        
        # Выбор команд
        self.setup_team_selection()
        
        # Кнопка прогноза
        self.setup_predict_button()
        
        # Область результатов
        self.setup_result_display()
        
        # Загрузка данных
        self.load_teams()

    def setup_mode_switch(self):
        """Настройка переключателя режимов"""
        self.mode_switch_layout = QHBoxLayout()
        self.data_mode_label = QLabel("Режим данных:")
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems(["Игровые данные", "Коэффициенты букмекеров"])
        self.data_mode_combo.currentIndexChanged.connect(self.switch_data_mode)
        
        self.mode_switch_layout.addWidget(self.data_mode_label)
        self.mode_switch_layout.addWidget(self.data_mode_combo)
        self.main_layout.addLayout(self.mode_switch_layout)

    def setup_odds_inputs(self):
        """Настройка полей ввода коэффициентов"""
        self.odds_layout = QHBoxLayout()
        self.odds_inputs = {
            'home': self.create_odds_input("Домашняя"),
            'draw': self.create_odds_input("Ничья"),
            'away': self.create_odds_input("Гостевая")
        }
        for widget in self.odds_inputs.values():
            self.odds_layout.addWidget(widget)
            widget.hide()
        self.main_layout.addLayout(self.odds_layout)

    def setup_team_selection(self):
        """Настройка выбора команд"""
        self.team_selection_layout = QHBoxLayout()
        
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
        
        self.team_selection_layout.addLayout(home_layout)
        self.team_selection_layout.addLayout(away_layout)
        self.main_layout.addLayout(self.team_selection_layout)

    def setup_predict_button(self):
        """Настройка кнопки прогноза"""
        self.predict_btn = QPushButton("Прогноз")
        self.predict_btn.clicked.connect(self.predict_match)
        self.predict_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-weight: bold; background-color: #4CAF50; color: white; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.main_layout.addWidget(self.predict_btn)

    def setup_result_display(self):
        """Настройка области результатов"""
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-family: monospace;")
        self.main_layout.addWidget(self.result_display)

    def create_odds_input(self, label):
        """Создает поле ввода коэффициента"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"{label} коэффициент:"))
        line_edit = QLineEdit()
        line_edit.setValidator(QDoubleValidator(1.0, 100.0, 2))
        line_edit.setPlaceholderText("1,75")
        layout.addWidget(line_edit)
        return widget
    
    def switch_data_mode(self, index):
        """Переключает между режимами ввода данных"""
        is_odds_mode = (index == 1)  # 1 - это Коэффициенты букмекеров
        
        # Показываем/скрываем соответствующие элементы
        self.home_combo.setVisible(not is_odds_mode)
        self.away_combo.setVisible(not is_odds_mode)
        
        for widget in self.odds_inputs.values():
            widget.setVisible(is_odds_mode)
        
        # Обновляем текст кнопки
        self.predict_btn.setText("Прогноз по коэффициентам" if is_odds_mode else "Прогноз по командам")
        
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
        """Обработчик нажатия кнопки прогноза"""
        if self.data_mode_combo.currentIndex() == 0:
            # Режим игровых данных (оригинальная логика)
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
        else:
            # Режим коэффициентов букмекеров
            try:
                # Получаем текст из полей ввода и заменяем запятые на точки
                home_odd_text = self.odds_inputs['home'].findChild(QLineEdit).text().replace(',', '.')
                draw_odd_text = self.odds_inputs['draw'].findChild(QLineEdit).text().replace(',', '.')
                away_odd_text = self.odds_inputs['away'].findChild(QLineEdit).text().replace(',', '.')
                
                # Преобразуем в числа
                home_odd = float(home_odd_text)
                draw_odd = float(draw_odd_text)
                away_odd = float(away_odd_text)
                
                if not all([home_odd, draw_odd, away_odd]):
                    raise ValueError("Все коэффициенты должны быть заполнены")
                    
                prediction = self.predictor_odds.predict_match(home_odd, draw_odd, away_odd)
                self.display_odds_prediction(prediction)
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Некорректные данные: {str(e)}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка прогноза", str(e))

    def display_odds_prediction(self, prediction):
        """Отображает прогноз по коэффициентам"""
        text = "=== Прогноз по коэффициентам ===\n\n"
        text += "Вероятности:\n"
        text += f"- Победа дома: {prediction['home_win']:.1%}\n"
        text += f"- Ничья: {prediction['draw']:.1%}\n"
        text += f"- Победа гостей: {prediction['away_win']:.1%}\n\n"
        
        rec = prediction['recommended']
        text += f"Рекомендуемая ставка: {rec['outcome']}\n"
        text += f"Вероятность: {rec['probability']:.1%}"
        
        self.result_display.setPlainText(text)

    def display_prediction(self, prediction):
        """Отображает результаты прогноза"""
        try:
            home = prediction['teams']['home']
            away = prediction['teams']['away']
            
            text = f"=== Прогноз на матч {home} vs {away} ===\n\n"
            
            # Данные Пуассона
            text += "=== Модель Пуассона ===\n"
            text += f"Ожидаемый счёт: {prediction['expected_score']['home']} - {prediction['expected_score']['away']}\n"
            text += "Вероятности:\n"
            text += f"- Победа {home}: {prediction['outcome_probabilities']['home_win']:.1%}\n"
            text += f"- Ничья: {prediction['outcome_probabilities']['draw']:.1%}\n"
            text += f"- Победа {away}: {prediction['outcome_probabilities']['away_win']:.1%}\n\n"
            
            # Данные Random Forest
            rf_pred = self.predictor.predict_with_rf(home, away)
            text += "=== Random Forest ===\n"
            text += f"Использована модель: {rf_pred['model_used']}\n"
            text += f"Тип матча: {'Близкий' if rf_pred['is_close_match'] else 'Обычный'}\n"
            text += "Вероятности:\n"
            text += f"- Победа {home}: {rf_pred['probabilities']['home_win']:.1%}\n"
            text += f"- Ничья: {rf_pred['probabilities']['draw']:.1%}\n"
            text += f"- Победа {away}: {rf_pred['probabilities']['away_win']:.1%}\n\n"
            
            # Усреднённый прогноз
            text += "=== Усреднённый прогноз ===\n"
            avg_probs = {
                'home_win': (prediction['outcome_probabilities']['home_win'] + rf_pred['probabilities']['home_win']) / 2,
                'draw': (prediction['outcome_probabilities']['draw'] + rf_pred['probabilities']['draw']) / 2,
                'away_win': (prediction['outcome_probabilities']['away_win'] + rf_pred['probabilities']['away_win']) / 2
            }
            
            max_outcome = max(avg_probs, key=avg_probs.get)
            outcomes_map = {
                'home_win': f'Победа {home}',
                'draw': 'Ничья',
                'away_win': f'Победа {away}'
            }
            
            text += "Средние вероятности:\n"
            for outcome, prob in avg_probs.items():
                text += f"- {outcomes_map[outcome]}: {prob:.1%}\n"
            
            if avg_probs[max_outcome] > 0.5:
                text += f"\nРекомендация: {outcomes_map[max_outcome]}"
            
            self.result_display.setPlainText(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить прогноз: {str(e)}")

    def translate_outcome(self, outcome_code):
        """Переводит коды исходов в текст"""
        translations = {
            'H': f'Победа {self.home_combo.currentText()}',
            'D': 'Ничья',
            'A': f'Победа {self.away_combo.currentText()}'
        }
        return translations.get(outcome_code, outcome_code)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Установка стиля  интерфейса
    app.setStyle("Fusion")
    
    window = FootballPredictorApp()
    window.show()
    sys.exit(app.exec())

