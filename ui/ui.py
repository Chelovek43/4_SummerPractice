import os
import pandas as pd
from PyQt6.QtCore import Qt, QSettings, QDate
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit,
    QMessageBox, QLineEdit, QTabWidget, QDateEdit
)
from PyQt6.QtGui import QDoubleValidator

from sklearn.model_selection import train_test_split

from core.predictor import FootballMatchPredictor
from core.odds_predictor import OddsMatchPredictor
from core.draw_binary import add_features
from core.draw_binary import DrawBinaryClassifier 
from ui.graphics_and_statictic import StatsGraphManager
from stats.statistics import StatisticsManager
from stats.graphics import GraphsManager

class FootballPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyOrg", "FootballPredictorApp")
        self.setup_predictors()
        self.init_ui()
        self.setup_window()
        self.restore_user_settings()

    ''' TODO: В идеале перенести в фоновое типа  
    class ModelLoaderThread(QThread):
        finished = pyqtSignal(object, object, object, object)
        def run(self):
    '''
    def setup_predictors(self):
        """Инициализация моделей предсказания с проверкой наличия файлов и обработкой ошибок"""
        try:
            if not os.path.exists("football.csv"):
                raise FileNotFoundError("Файл football.csv не найден.")
            self.predictor = FootballMatchPredictor("football.csv")
            self.stats_manager = StatisticsManager(df=self.predictor.df)
            self.graphs_manager = GraphsManager(self.stats_manager, self)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки данных", f"Ошибка при загрузке football.csv: {str(e)}")
            raise

        try:
            if not os.path.exists("Laliga.csv"):
                raise FileNotFoundError("Файл Laliga.csv не найден.")
            self.odds_predictor = OddsMatchPredictor("Laliga.csv")
            self.odds_predictor.train_model()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки данных", f"Ошибка при загрузке Laliga.csv: {str(e)}")
            raise

        features = [
        "HomeForm", "AwayForm", "HomeAttack", "AwayDefense",
        "HeadToHeadWinRate", "HomeLast3Goals", "AwayLast3Conceded"
        ]
        df_fit = self.predictor.df.dropna(subset=features + ["FTR"])
        X = df_fit[features]
        y = (df_fit["FTR"] == "D").astype(int)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.draw_clf = DrawBinaryClassifier(features=features)
        self.draw_clf.fit(df_fit, X_val=X_val, y_val=y_val, autotune_threshold=True, plot=True)

    def setup_window(self):
        """Настройка основного окна"""
        self.setWindowTitle("Football Match Predictor")
        self.setMinimumSize(600, 500)

    def init_ui(self):
        """Инициализация пользовательского интерфейса"""

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout (для всего окна)
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
        
        # Добавляем элементы управления для режима прогноза
        self.setup_mode_switch()      # Переключатель между игровыми данными и коэффициентами
        self.setup_odds_inputs()      # Поля ввода коэффициентов
        self.setup_team_selection()   # Выбор команд
        self.setup_date_selection()   # Выбор даты
        self.setup_predict_button()   # Кнопка "Прогноз"
        self.load_teams()             # Загрузка списка команд
        self.setup_result_display()   # Область вывода результата
        
        # Вкладка 2: Аналитика (статистика и сравнение команд)
        self.tab_analytics = StatsGraphManager(
            self.stats_manager, 
            self,
            show_team_select=True
        )
        self.tabs.addTab(self.tab_analytics, "Аналитика")
        self.tab_analytics.show_stats()  # Показываем статистику по умолчанию
        
        # Вкладка 3: Графики (тоже с выбором команд)
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
        """
        Обработчик переключения вкладок.
        При смене вкладки обновляет статистику или графики для выбранных команд.
        """
        print(f"Переключено на вкладку {index + 1}")

        if index == 0:  # Прогноз
            """
            Не требует автоматического обновления данных при переключении, потому что:
            Все элементы (команды, коэффициенты, режим) уже синхронизированы через другие обработчики.
            Прогноз строится только по нажатию кнопки "Прогноз".
            Нет необходимости что-то обновлять автоматически при каждом переходе на эту вкладку.
            """
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
        """Настройка заголовка приложения (отображается в верхней части окна)"""

        title = QLabel("Football Match Predictor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            margin-bottom: 20px;
        """)
        self.main_layout.addWidget(title)

    def setup_mode_switch(self):
        """Настройка переключателя режимов (игровые данные / коэффициенты букмекеров)"""

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Режим данных:"))
        
        # Создаем выпадающий список для выбора режима работы
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems([
            "Игровые данные", 
            "Коэффициенты букмекеров"
        ])
        # При смене режима вызывается обработчик switch_data_mode
        self.data_mode_combo.currentIndexChanged.connect(self.switch_data_mode)
        
        layout.addWidget(self.data_mode_combo)
        self.predict_layout.addLayout(layout)

    def setup_odds_inputs(self):
        """Настройка полей ввода коэффициентов для прогноза по букмекерским данным"""

        self.odds_layout = QHBoxLayout()
        self.odds_inputs = {
            'home': self.create_odds_input("Домашняя"),
            'draw': self.create_odds_input("Ничья"),
            'away': self.create_odds_input("Гостевая")
        }
        
        # Добавляем поля в layout и скрываем их по умолчанию
        for widget in self.odds_inputs.values():
            self.odds_layout.addWidget(widget)
            widget.hide()
            
        self.predict_layout.addLayout(self.odds_layout)

    def create_odds_input(self, label):
        """Создание поля ввода коэффициента с подписью"""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"{label} коэффициент:"))
        
        # Поле ввода с валидатором для чисел с плавающей точкой
        line_edit = QLineEdit()
        line_edit.setValidator(QDoubleValidator(1.0, 100.0, 2))
        line_edit.setPlaceholderText("1,75")
        
        layout.addWidget(line_edit)
        return widget

    def setup_team_selection(self):
        """Настройка выбора команд (домашняя и гостевая) для прогноза"""

        layout = QHBoxLayout()
        
        # Создаем комбобоксы напрямую 
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
        """Синхронизация выбора команд между вкладками с минимизацией лишних сигналов"""

        try:
            # Обновляем значения в комбобоксах, если явно переданы новые значения и они отличаются
            if home_team and self.home_combo.currentText() != home_team:
                self.home_combo.setCurrentText(home_team)
            if away_team and self.away_combo.currentText() != away_team:
                self.away_combo.setCurrentText(away_team)
            
            # Получаем актуальные значения
            current_home = home_team if home_team else self.home_combo.currentText()
            current_away = away_team if away_team else self.away_combo.currentText()

            # Проверяем, что вкладки уже созданы
            if not hasattr(self, 'tab_analytics') or not hasattr(self, 'tab_graphs'):
                return
            
            # Проверка на одинаковые команды 
            if current_home == current_away:
                if not getattr(self, "_same_team_warning_shown", False):
                    self._same_team_warning_shown = True
                    QMessageBox.warning(self, "Ошибка", "Команды не должны быть одинаковыми!")
                return
            self._same_team_warning_shown = False
            
            # Синхронизируем все вкладки
            for tab in [self.tab_analytics, self.tab_graphs]:
                if tab and hasattr(tab, 'home_combo'):
                    if tab.home_combo.currentText() != current_home:
                        tab.home_combo.blockSignals(True)
                        tab.home_combo.setCurrentText(current_home)
                        tab.home_combo.blockSignals(False)
                        
                if tab and hasattr(tab, 'away_combo'):
                    if tab.away_combo.currentText() != current_away:
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

    def setup_date_selection(self):
        """
        Добавляет поле для выбора даты матча с календарём (QDateEdit).
        """
        layout = QHBoxLayout()
        label = QLabel("Дата матча:")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setToolTip("Выберите дату матча (по умолчанию — сегодня)")

        layout.addWidget(label)
        layout.addWidget(self.date_edit)
        layout.addStretch()
        self.predict_layout.addLayout(layout)

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

        self.home_combo.blockSignals(True)
        self.away_combo.blockSignals(True)

        self.home_combo.clear()
        self.away_combo.clear()
        self.home_combo.addItems(home_teams)
        self.away_combo.addItems(away_teams)

        if hasattr(self, 'tab_analytics'):
            if hasattr(self.tab_analytics, 'home_combo'):
                self.tab_analytics.home_combo.clear()
                self.tab_analytics.home_combo.addItems(home_teams)
            if hasattr(self.tab_analytics, 'away_combo'):
                self.tab_analytics.away_combo.clear()
                self.tab_analytics.away_combo.addItems(away_teams)

        # Гарантируем, что команд больше одной
        self.home_combo.setCurrentIndex(0)
        self.away_combo.setCurrentIndex(1)

        self.home_combo.blockSignals(False)
        self.away_combo.blockSignals(False)

        self.sync_team_selection(
            home_team=self.home_combo.currentText(),
            away_team=self.away_combo.currentText()
        )
        
        
    def switch_data_mode(self, index):
        """
        Переключение между режимами ввода данных (игровые данные/коэффициенты)
        Показывает или скрывает соответствующие элементы интерфейса.
        """

        is_odds_mode = (index == 1)
        
        # Скрываем или показываем комбобоксы команд
        self.home_combo.setVisible(not is_odds_mode)
        self.away_combo.setVisible(not is_odds_mode)

        # Показываем или скрываем поля ввода коэффициентов
        for widget in self.odds_inputs.values():
            widget.setVisible(is_odds_mode)
        
        # Меняем текст на кнопке в зависимости от режима
        self.predict_btn.setText(
            "Прогноз по коэффициентам" if is_odds_mode 
            else "Прогноз по командам"
        )

    def predict_match(self):
        """Обработка прогноза: вызывает нужный метод в зависимости от выбранного режима"""
        if self.data_mode_combo.currentIndex() == 0:
            self.predict_by_teams()
        else:
            self.predict_by_odds()

    def predict_by_teams(self):
        home_team = self.home_combo.currentText()
        away_team = self.away_combo.currentText()
        if home_team == away_team:
            QMessageBox.warning(self, "Ошибка", "Команды не должны быть одинаковыми!")
            return

        match_qdate = self.date_edit.date()
        match_date = match_qdate.toString("yyyy-MM-dd") if match_qdate else None

        try:
            rf_result = self.predictor.predict_with_rf(home_team, away_team, match_date=match_date)
            poisson_result = self.predictor.predict_match(home_team, away_team, match_date=match_date)
            combined_result = self.predictor.combined_predict(
                self.draw_clf, home_team, away_team, match_date
            )

            text_blocks = [
                self.display_prediction(poisson_result, match_date=match_date),
                "",
                self.display_combined_prediction(combined_result, home_team, away_team, match_date)
            ]
            self.result_display.setPlainText("\n".join(text_blocks))

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
        """Получение значений коэффициентов из полей ввода"""

        odds_texts = [
            self.odds_inputs[key].findChild(QLineEdit).text().replace(',', '.')
            for key in ['home', 'draw', 'away']
        ]
        
        # Проверяем, что все поля заполнены
        if not all(odds_texts):
            raise ValueError("Все коэффициенты должны быть заполнены")
            
        return [float(odd) for odd in odds_texts]
    
    def get_rf_prediction_text(self, home, away, rf_pred):
        """
        Получение текста прогноза Random Forest
        Запрашивает у предсказателя результат RF и формирует текстовый блок для вывода
        """

        
        return "\n".join([
            "=== Random Forest ===",
            f"Использована модель: {rf_pred['model_used']}",
            f"Тип матча: {'Близкий' if rf_pred['is_close_match'] else 'Обычный'}",
            "Вероятности:",
            f"- Победа {home}: {rf_pred['probabilities']['home_win']:.1%}",
            f"- Ничья: {rf_pred['probabilities']['draw']:.1%}",
            f"- Победа {away}: {rf_pred['probabilities']['away_win']:.1%}"
        ])

    def get_average_prediction_text(self, prediction, home, away, rf_pred):
        """
        Получение текста усредненного прогноза

        Усредняет вероятности исходов между моделью Пуассона и Random Forest,
        формирует текстовый блок с рекомендацией, если вероятность превышает 50%
        """

        
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


    # TODO: кэшировать типа @lru_cache(maxsize=128)  
    def display_prediction(self, prediction, match_date=None):
        """
        Отображение прогноза по командам (игровые данные)

        Формирует и выводит подробный текстовый отчет по прогнозу:
        - результат модели Пуассона,
        - результат Random Forest,
        - усреднённый прогноз.
        """

        try:
            home = prediction['teams']['home']
            away = prediction['teams']['away']
            rf_pred = self.predictor.predict_with_rf(home, away, match_date=match_date)
            date_str = f" на дату {match_date}" if match_date else ""
            text = [
                f"=== Прогноз на матч {home} vs {away}{date_str} ===",
                "",
                "=== Модель Пуассона ===",
                f"Ожидаемый счёт: {prediction['expected_score']['home']} - {prediction['expected_score']['away']}",
                "Вероятности:",
                f"- Победа {home}: {prediction['outcome_probabilities']['home_win']:.1%}",
                f"- Ничья: {prediction['outcome_probabilities']['draw']:.1%}",
                f"- Победа {away}: {prediction['outcome_probabilities']['away_win']:.1%}",
                "",
                self.get_rf_prediction_text(home, away, rf_pred),
                "",
                self.get_average_prediction_text(prediction, home, away, rf_pred)
            ]
            return "\n".join(text)
        except Exception as e:
            return f"Ошибка: {str(e)}"

    def display_odds_prediction(self, prediction):
        """
        Отображение прогноза по коэффициентам

        Формирует и выводит текстовый отчет по вероятностям исходов и рекомендуемой ставке
        на основе модели, обученной на букмекерских коэффициентах
        """

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

    def display_combined_prediction(self, result, home, away, match_date=None):
        date_str = f" на дату {match_date}" if match_date else ""
        lines = [
            f"=== Комбинированный прогноз на матч {home} vs {away}{date_str} ===",
            f"Вероятность ничьей по бинарному классификатору: {result['draw_proba']:.2%}"
        ]
        rf_probs = result.get('rf_probabilities')
        return "\n".join(lines)

    # Тест настроек сохранения выбора
    def restore_user_settings(self):
        last_home = self.settings.value("last_home_team", "")
        last_away = self.settings.value("last_away_team", "")
        last_mode = int(self.settings.value("last_data_mode", 0))
        last_tab = int(self.settings.value("last_tab", 0))
        if last_home:
            self.home_combo.setCurrentText(last_home)
        if last_away:
            self.away_combo.setCurrentText(last_away)
        self.data_mode_combo.setCurrentIndex(last_mode)
        self.tabs.setCurrentIndex(last_tab)

    def closeEvent(self, event):
        self.settings.setValue("last_home_team", self.home_combo.currentText())
        self.settings.setValue("last_away_team", self.away_combo.currentText())
        self.settings.setValue("last_data_mode", self.data_mode_combo.currentIndex())
        self.settings.setValue("last_tab", self.tabs.currentIndex())
        super().closeEvent(event)