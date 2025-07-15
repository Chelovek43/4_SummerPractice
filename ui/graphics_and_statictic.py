from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QComboBox, QHBoxLayout
from PyQt6.QtCore import Qt
from stats.statistics import StatisticsManager

class StatsGraphManager(QWidget):
    def __init__(self, stats_manager, parent=None, show_team_select=True):
        super().__init__(parent)
        self.parent = parent
        self.stats_manager = stats_manager
        self.show_team_select = show_team_select  # Флаг для отображения выбора команд
        self.current_period = 'all'  # текущий выбранный период
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Добавляем выбор команд только если нужно
        if self.show_team_select:
            self.setup_team_selection()
        self.setup_period_selection()
        self.setup_stats_display()
        self.setup_graphs_display()

    def setup_period_selection(self):
        """Настройка выбора периода статистики"""
        period_layout = QHBoxLayout()
        
        period_label = QLabel("Период статистики:")
        self.period_combo = QComboBox()
        self.period_combo.addItem("Вся статистика", "all")
        self.period_combo.addItem("Последние 5 матчей", "5")
        self.period_combo.addItem("Последние 10 матчей", "10")
        self.period_combo.addItem("Весь сезон (38 матчей)", "38")
        self.period_combo.addItem("Только между этими командами", "h2h")
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        
        period_layout.addWidget(period_label)
        period_layout.addWidget(self.period_combo)
        period_layout.addStretch()  # Добавляем растягивающееся пространство
        
        self.layout.addLayout(period_layout)


    def on_period_changed(self):
        """Обработчик изменения выбранного периода статистики"""
        self.current_period = self.period_combo.currentData()
        self.refresh_stats() 

    def setup_stats_display(self):
        """Настройка отображения статистики"""
        self.stats_title = QLabel("Статистика команд")
        self.stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_title.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.stats_title)
        
        self.stats_container = QHBoxLayout()
        self.layout.addLayout(self.stats_container)
        
        # Колонки для команд
        self.team1_column = QVBoxLayout()
        self.team1_label = QLabel("Команда 1")
        self.team1_stats = QTextEdit()
        self.team1_stats.setReadOnly(True)
        self.team1_column.addWidget(self.team1_label)
        self.team1_column.addWidget(self.team1_stats)
        
        self.team2_column = QVBoxLayout()
        self.team2_label = QLabel("Команда 2")
        self.team2_stats = QTextEdit()
        self.team2_stats.setReadOnly(True)
        self.team2_column.addWidget(self.team2_label)
        self.team2_column.addWidget(self.team2_stats)
        
        self.stats_container.addLayout(self.team1_column)
        self.stats_container.addLayout(self.team2_column)

    def setup_graphs_display(self):
        """Настройка отображения графиков"""
        self.graphs_display = QTextEdit()
        self.graphs_display.setReadOnly(True)
        self.layout.addWidget(self.graphs_display)
        self.graphs_display.hide()

    def update_stats(self, team1, team2):
        """Обновление статистики с учетом выбранного периода"""
        self.team1_name = team1
        self.team2_name = team2
        self.refresh_stats()

    def refresh_stats(self):
        """Обновление отображения статистики (использует текущие команды и период)"""
        if not hasattr(self, 'team1_name') or not hasattr(self, 'team2_name'):
            return
        
        print(f"\nRefreshing stats with period: {self.current_period}")
        
        stats1 = self.stats_manager.get_team_stats(
            self.team1_name, 
            period=str(self.current_period),  # убедитесь, что передается строка
            opponent=self.team2_name if self.current_period == 'h2h' else None
        )
        
        stats2 = self.stats_manager.get_team_stats(
            self.team2_name,
            period=str(self.current_period),
            opponent=self.team1_name if self.current_period == 'h2h' else None
        )
        
        print("Team 1 stats:", stats1)
        print("Team 2 stats:", stats2)
        
        self.team1_stats.setPlainText(self.format_stats(stats1))
        self.team2_stats.setPlainText(self.format_stats(stats2))

    def format_stats(self, stats):
        if not stats:
            return "Статистика недоступна"
        
        try:
            # Преобразуем numpy типы в стандартные Python типы
            matches = int(stats.get('matches', 0))
            wins = int(stats.get('wins', 0))
            win_rate = float(stats.get('win_rate', 0))
            form = float(stats.get('form', 0))
            
            return (
                f"Матчи: {matches}\n"
                f"Победы: {wins}\n"
                f"Процент побед: {win_rate:.2f}%\n"
                f"Форма: {form:.2f}"
            )
        except Exception as e:
            return f"Ошибка форматирования статистики: {str(e)}"
        
    def show_stats(self):
        """Показать статистику"""
        self.stats_title.show()
        self.team1_label.show()
        self.team1_stats.show()
        self.team2_label.show()
        self.team2_stats.show()
        self.graphs_display.hide()

    def show_graphs(self):
        """Показать графики"""
        self.stats_title.hide()
        self.team1_label.hide()
        self.team1_stats.hide()
        self.team2_label.hide()
        self.team2_stats.hide()
        self.graphs_display.show()

    def setup_team_selection(self):
        """Настройка выбора команд с синхронизацией"""
        if not self.parent or not hasattr(self.parent, 'home_combo'):
            return
            
        layout = QHBoxLayout()
        
        # Создаём комбобоксы
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        
        # Копируем данные из родителя
        self.home_combo.addItems([self.parent.home_combo.itemText(i) 
                                for i in range(self.parent.home_combo.count())])
        self.away_combo.addItems([self.parent.away_combo.itemText(i) 
                                for i in range(self.parent.away_combo.count())])
        
        # Устанавливаем текущие значения
        self.home_combo.setCurrentText(self.parent.home_combo.currentText())
        self.away_combo.setCurrentText(self.parent.away_combo.currentText())
        
        # Настройка layout 
        home_layout = QVBoxLayout()
        home_layout.addWidget(QLabel("Домашняя команда:"))
        home_layout.addWidget(self.home_combo)
        
        away_layout = QVBoxLayout()
        away_layout.addWidget(QLabel("Гостевая команда:"))
        away_layout.addWidget(self.away_combo)
        
        layout.addLayout(home_layout)
        layout.addLayout(away_layout)
        self.layout.addLayout(layout)
        
        # Подключаем сигналы напрямую к родительскому обработчику
        self.home_combo.currentTextChanged.connect(
            lambda: self.parent.sync_team_selection(home_team=self.home_combo.currentText()))
        self.away_combo.currentTextChanged.connect(
            lambda: self.parent.sync_team_selection(away_team=self.away_combo.currentText()))
