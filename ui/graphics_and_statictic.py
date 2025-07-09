from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QComboBox, QHBoxLayout
from PyQt6.QtCore import Qt
from stats.statistics import StatisticsManager

class StatsGraphManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Выбор команд 
        self.setup_team_selection()
        
        # Заголовок статистики
        self.stats_title = QLabel("Статистика команд")
        self.stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_title.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.stats_title)
        
        # Двухколоночный контейнер для статистики
        self.stats_container = QHBoxLayout()
        self.layout.addLayout(self.stats_container)
        
        # Колонка для Команды 1 (Домашняя)
        self.team1_column = QVBoxLayout()
        self.team1_label = QLabel("Команда 1")
        self.team1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.team1_stats = QTextEdit()
        self.team1_stats.setReadOnly(True)
        
        self.team1_column.addWidget(self.team1_label)
        self.team1_column.addWidget(self.team1_stats)
        self.stats_container.addLayout(self.team1_column)
        
        # Колонка для Команды 2 (Гостевая)
        self.team2_column = QVBoxLayout()
        self.team2_label = QLabel("Команда 2")
        self.team2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.team2_stats = QTextEdit()
        self.team2_stats.setReadOnly(True)
        
        self.team2_column.addWidget(self.team2_label)
        self.team2_column.addWidget(self.team2_stats)
        self.stats_container.addLayout(self.team2_column)
        
        # Поле для графиков (изначально скрыто)
        self.graphs_display = QTextEdit()
        self.graphs_display.setReadOnly(True)
        self.layout.addWidget(self.graphs_display)
        self.graphs_display.hide()
        
        # Показываем тестовые данные
        self.show_test_stats()
        
    def show_test_stats(self):
        """Показать тестовую статистику"""
        self.team1_stats.setPlainText(
            "Матчи: 10\n"
            "Победы: 5 (50%)\n"
            "Форма: 1.75"
        )
        self.team2_stats.setPlainText(
            "Матчи: 12\n"
            "Победы: 6 (50%)\n"
            "Форма: 1.80"
        )

    def update_stats(self, team1, team2):
        """Обновить статистику для реальных команд"""
        self.team1_label.setText(team1)
        self.team2_label.setText(team2)
        
        # Здесь будет логика загрузки реальных данных
        self.show_test_stats()  # Пока используем тестовые данные

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
