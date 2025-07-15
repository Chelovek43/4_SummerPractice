from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit,
    QMessageBox, QLineEdit, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

from core.predicrtor import FootballMatchPredictor
from core.odds_predicror import OddsMatchPredictor
from ui.graphics_and_statictic import StatsGraphManager
from stats.statistics import StatisticsManager

class FootballPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.tab_analytics = None
        self.tab_graphs = None
        self.stats_manager = StatisticsManager()
        self.stats_manager.load_data("football.csv")
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
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout (теперь для всего окна)
        self.main_layout = QVBoxLayout(central_widget)
        
        # Добавляем заголовок 
        self.setup_title()
        
        # Создаем виджет вкладок
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Вкладка 1: Прогноз 
        self.tab_predict = QWidget()
        self.tabs.addTab(self.tab_predict, "Прогноз")
        
        # Создаем отдельный layout для вкладки прогноза
        self.predict_layout = QVBoxLayout(self.tab_predict)
        
        self.setup_mode_switch()
        self.setup_odds_inputs()
        self.setup_team_selection()
        self.setup_predict_button()
        self.setup_result_display()
        self.load_teams()
        
        self.tab_analytics = StatsGraphManager(
            self.stats_manager, 
            self,
            show_team_select=True
        )
        self.tabs.addTab(self.tab_analytics, "Аналитика")
        self.tab_analytics.show_stats()  # Показываем статистику по умолчанию
        
        # Вкладка Графики (тоже с выбором команд)
        self.tab_graphs = StatsGraphManager(
            self.stats_manager,
            self,
            show_team_select=True
        )
        self.tabs.addTab(self.tab_graphs, "Графики")
        self.tab_graphs.show_graphs()  # Показываем графики по умолчанию

        
        # Подключаем обработчик смены вкладок
        self.tabs.currentChanged.connect(self.on_tab_changed)


    def on_tab_changed(self, index):
        """Обработчик переключения вкладок"""
        print(f"Переключено на вкладку {index + 1}")

        if index == 0:  # Прогноз
            pass
        elif index == 1:  # Аналитика
            home = self.home_combo.currentText()
            away = self.away_combo.currentText()
            self.tab_analytics.update_stats(home, away)
        elif index == 2:  # Графики
            home = self.home_combo.currentText()
            away = self.away_combo.currentText()
            self.tab_graphs.update_stats(home, away)


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
        self.predict_layout.addLayout(layout)

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
            
        self.predict_layout.addLayout(self.odds_layout)

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
        """Настройка выбора команд (исправленная версия)"""
        layout = QHBoxLayout()
        
        # Создаем комбобоксы напрямую (без create_team_combo)
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        
        # Добавляем подписи
        home_label = QLabel("Домашняя команда:")
        away_label = QLabel("Гостевая команда:")
        
        # Группируем в вертикальные layout
        home_layout = QVBoxLayout()
        home_layout.addWidget(home_label)
        home_layout.addWidget(self.home_combo)
        
        away_layout = QVBoxLayout()
        away_layout.addWidget(away_label)
        away_layout.addWidget(self.away_combo)
        
        # Добавляем в основной layout
        layout.addLayout(home_layout)
        layout.addLayout(away_layout)
        self.predict_layout.addLayout(layout)
        
        # Подключаем сигналы синхронизации
        self.home_combo.currentTextChanged.connect(
    lambda: self.sync_team_selection(home_team=self.home_combo.currentText()))
        self.away_combo.currentTextChanged.connect(
    lambda: self.sync_team_selection(away_team=self.away_combo.currentText()))
    
    def sync_team_selection(self, home_team=None, away_team=None):
        """Синхронизация выбора команд между вкладками
        Args:
            home_team (str, optional): Если None - берёт из текущего комбобокса
            away_team (str, optional): Если None - берёт из текущего комбобокса
        """
        try:
            # Обновляем комбобоксы
            if home_team:
                self.home_combo.setCurrentText(home_team)
            if away_team:
                self.away_combo.setCurrentText(away_team)
            
            # Получаем актуальные значения
            current_home = home_team if home_team else self.home_combo.currentText()
            current_away = away_team if away_team else self.away_combo.currentText()
            
            # Синхронизируем все вкладки
            for tab in [self.tab_analytics, self.tab_graphs]:
                if tab and hasattr(tab, 'home_combo'):
                    tab.home_combo.blockSignals(True)
                    tab.home_combo.setCurrentText(current_home)
                    tab.home_combo.blockSignals(False)
                    
                if tab and hasattr(tab, 'away_combo'):
                    tab.away_combo.blockSignals(True)
                    tab.away_combo.setCurrentText(current_away)
                    tab.away_combo.blockSignals(False)
                
                # Обновляем данные
                if tab == self.tab_analytics:
                    tab.update_stats(current_home, current_away)
                elif tab == self.tab_graphs:
                    tab.update_stats(current_home, current_away)
                        
        except Exception as e:
            print(f"Ошибка синхронизации: {e}")

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
        self.predict_layout.addWidget(self.predict_btn)

    def setup_result_display(self):
        """Настройка области результатов"""
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-family: monospace;")
        self.predict_layout.addWidget(self.result_display)

    def load_teams(self):
        """Загружает команды во все компоненты"""
        home_teams, away_teams = self.predictor.get_team_list()
        
        # Основные комбобоксы
        self.home_combo.clear()
        self.away_combo.clear()
        self.home_combo.addItems(home_teams)
        self.away_combo.addItems(away_teams)
        
        # Синхронизация аналитики
        if hasattr(self, 'tab_analytics'):
            if hasattr(self.tab_analytics, 'home_combo'):
                self.tab_analytics.home_combo.clear()
                self.tab_analytics.home_combo.addItems(home_teams)
                
            if hasattr(self.tab_analytics, 'away_combo'):
                self.tab_analytics.away_combo.clear()
                self.tab_analytics.away_combo.addItems(away_teams)
        
        # Установка начальных значений
        if home_teams:
            self.home_combo.setCurrentIndex(0)
        if away_teams:
            self.away_combo.setCurrentIndex(0 if len(away_teams) == 1 else 1)
        
        
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

        