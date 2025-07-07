from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit,
    QMessageBox, QLineEdit, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

from core.predicrtor import FootballMatchPredictor
from core.odds_predicror import OddsMatchPredictor


class FootballPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_predictors()
        self.init_ui()
        self.setup_window()
        
    def setup_predictors(self):
        """Инициализация моделей предсказания"""
        self.predictor = FootballMatchPredictor("football.csv")
        self.odds_predictor = OddsMatchPredictor("Laliga.csv")
        self.odds_predictor.train_model()

    def setup_window(self):
        """Настройка основного окна"""
        self.setWindowTitle("Football Match Predictor")
        self.setMinimumSize(600, 500)

    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.main_layout = QVBoxLayout(central_widget)
        self.setup_title()
        self.setup_mode_switch()
        self.setup_odds_inputs()
        self.setup_team_selection()
        self.setup_predict_button()
        self.setup_result_display()
        self.load_teams()


        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Вкладка 1: Прогноз
        tab1 = QWidget()
        self.tabs.addTab(tab1, "Прогноз")
        layout1 = QVBoxLayout(tab1)
        self.label1 = QLabel("Здесь будет форма прогноза")
        layout1.addWidget(self.label1)
        
        # Вкладка 2: Статистика
        tab2 = QWidget()
        self.tabs.addTab(tab2, "Статистика")
        layout2 = QVBoxLayout(tab2)
        self.label2 = QLabel("Здесь будет статистика команд")
        layout2.addWidget(self.label2)
        
        # Вкладка 3: Графики
        tab3 = QWidget()
        self.tabs.addTab(tab3, "Графики")
        layout3 = QVBoxLayout(tab3)
        self.label3 = QLabel("Здесь будут графики анализа")
        layout3.addWidget(self.label3)
        


    def setup_title(self):
        """Настройка заголовка"""
        title = QLabel("Football Match Predictor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            margin-bottom: 20px;
        """)
        self.main_layout.addWidget(title)

    def setup_mode_switch(self):
        """Настройка переключателя режимов"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Режим данных:"))
        
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems([
            "Игровые данные", 
            "Коэффициенты букмекеров"
        ])
        self.data_mode_combo.currentIndexChanged.connect(self.switch_data_mode)
        
        layout.addWidget(self.data_mode_combo)
        self.main_layout.addLayout(layout)

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

    def create_odds_input(self, label):
        """Создание поля ввода коэффициента"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"{label} коэффициент:"))
        
        line_edit = QLineEdit()
        line_edit.setValidator(QDoubleValidator(1.0, 100.0, 2))
        line_edit.setPlaceholderText("1,75")
        
        layout.addWidget(line_edit)
        return widget

    def setup_team_selection(self):
        """Настройка выбора команд"""
        layout = QHBoxLayout()
        
        self.home_combo = self.create_team_combo("Домашняя команда:")
        self.away_combo = self.create_team_combo("Гостевая команда:")
        
        layout.addLayout(self.home_combo)
        layout.addLayout(self.away_combo)
        self.main_layout.addLayout(layout)

    def create_team_combo(self, label):
        """Создание выпадающего списка команд"""
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        
        combo = QComboBox()
        layout.addWidget(combo)
        
        return layout

    def setup_predict_button(self):
        """Настройка кнопки прогноза"""
        self.predict_btn = QPushButton("Прогноз")
        self.predict_btn.clicked.connect(self.predict_match)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; 
                font-weight: bold; 
                background-color: #4CAF50; 
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.main_layout.addWidget(self.predict_btn)

    def setup_result_display(self):
        """Настройка области результатов"""
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-family: monospace;")
        self.main_layout.addWidget(self.result_display)

    def load_teams(self):
        """Загрузка списков команд"""
        home_teams, away_teams = self.predictor.get_team_list()
        
        self.home_combo.itemAt(1).widget().clear()
        self.away_combo.itemAt(1).widget().clear()
        
        self.home_combo.itemAt(1).widget().addItems(home_teams)
        self.away_combo.itemAt(1).widget().addItems(away_teams)
        
        if home_teams:
            self.home_combo.itemAt(1).widget().setCurrentIndex(0)
        if away_teams:
            self.away_combo.itemAt(1).widget().setCurrentIndex(0 if len(away_teams) == 1 else 1)

    def switch_data_mode(self, index):
        """Переключение между режимами ввода данных"""
        is_odds_mode = (index == 1)
        
        self.home_combo.itemAt(1).widget().setVisible(not is_odds_mode)
        self.away_combo.itemAt(1).widget().setVisible(not is_odds_mode)
        
        for widget in self.odds_inputs.values():
            widget.setVisible(is_odds_mode)
        
        self.predict_btn.setText(
            "Прогноз по коэффициентам" if is_odds_mode 
            else "Прогноз по командам"
        )

    def predict_match(self):
        """Обработка прогноза"""
        if self.data_mode_combo.currentIndex() == 0:
            self.predict_by_teams()
        else:
            self.predict_by_odds()

    def predict_by_teams(self):
        """Прогноз по командам"""
        home_team = self.home_combo.itemAt(1).widget().currentText()
        away_team = self.away_combo.itemAt(1).widget().currentText()
        
        if home_team == away_team:
            QMessageBox.warning(self, "Ошибка", "Команды не должны быть одинаковыми!")
            return
        
        try:
            prediction = self.predictor.predict_match(home_team, away_team)
            self.display_prediction(prediction)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка прогноза: {str(e)}")

    def predict_by_odds(self):
        """Прогноз по коэффициентам"""
        try:
            odds = self.get_odds_values()
            prediction = self.odds_predictor.predict_match(*odds)
            self.display_odds_prediction(prediction)
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Некорректные данные: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка прогноза", str(e))

    def get_odds_values(self):
        """Получение значений коэффициентов"""
        odds_texts = [
            self.odds_inputs[key].findChild(QLineEdit).text().replace(',', '.')
            for key in ['home', 'draw', 'away']
        ]
        
        if not all(odds_texts):
            raise ValueError("Все коэффициенты должны быть заполнены")
            
        return [float(odd) for odd in odds_texts]

    def display_prediction(self, prediction):
        """Отображение прогноза по командам"""
        try:
            home = prediction['teams']['home']
            away = prediction['teams']['away']
            
            text = [
                f"=== Прогноз на матч {home} vs {away} ===",
                "",
                "=== Модель Пуассона ===",
                f"Ожидаемый счёт: {prediction['expected_score']['home']} - {prediction['expected_score']['away']}",
                "Вероятности:",
                f"- Победа {home}: {prediction['outcome_probabilities']['home_win']:.1%}",
                f"- Ничья: {prediction['outcome_probabilities']['draw']:.1%}",
                f"- Победа {away}: {prediction['outcome_probabilities']['away_win']:.1%}",
                "",
                self.get_rf_prediction_text(home, away),
                "",
                self.get_average_prediction_text(prediction, home, away)
            ]
            
            self.result_display.setPlainText("\n".join(text))
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить прогноз: {str(e)}")

    def get_rf_prediction_text(self, home, away):
        """Получение текста прогноза Random Forest"""
        rf_pred = self.predictor.predict_with_rf(home, away)
        return "\n".join([
            "=== Random Forest ===",
            f"Использована модель: {rf_pred['model_used']}",
            f"Тип матча: {'Близкий' if rf_pred['is_close_match'] else 'Обычный'}",
            "Вероятности:",
            f"- Победа {home}: {rf_pred['probabilities']['home_win']:.1%}",
            f"- Ничья: {rf_pred['probabilities']['draw']:.1%}",
            f"- Победа {away}: {rf_pred['probabilities']['away_win']:.1%}"
        ])

    def get_average_prediction_text(self, prediction, home, away):
        """Получение текста усредненного прогноза"""
        rf_pred = self.predictor.predict_with_rf(home, away)
        avg_probs = {
            'home_win': (prediction['outcome_probabilities']['home_win'] + rf_pred['probabilities']['home_win']) / 2,
            'draw': (prediction['outcome_probabilities']['draw'] + rf_pred['probabilities']['draw']) / 2,
            'away_win': (prediction['outcome_probabilities']['away_win'] + rf_pred['probabilities']['away_win']) / 2
        }
        
        outcomes_map = {
            'home_win': f'Победа {home}',
            'draw': 'Ничья',
            'away_win': f'Победа {away}'
        }
        
        max_outcome = max(avg_probs, key=avg_probs.get)
        lines = [
            "=== Усреднённый прогноз ===",
            "Средние вероятности:"
        ]
        
        lines.extend(
            f"- {outcomes_map[outcome]}: {prob:.1%}"
            for outcome, prob in avg_probs.items()
        )
        
        if avg_probs[max_outcome] > 0.5:
            lines.append(f"\nРекомендация: {outcomes_map[max_outcome]}")
            
        return "\n".join(lines)

    def display_odds_prediction(self, prediction):
        """Отображение прогноза по коэффициентам"""
        text = [
            "=== Прогноз по коэффициентам ===",
            "",
            "Вероятности:",
            f"- Победа дома: {prediction['home_win']:.1%}",
            f"- Ничья: {prediction['draw']:.1%}",
            f"- Победа гостей: {prediction['away_win']:.1%}",
            "",
            f"Рекомендуемая ставка: {prediction['recommended']['outcome']}",
            f"Вероятность: {prediction['recommended']['probability']:.1%}"
        ]
        
        self.result_display.setPlainText("\n".join(text))