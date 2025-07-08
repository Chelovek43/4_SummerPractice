from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QComboBox, QHBoxLayout
from PyQt6.QtCore import Qt

class StatsGraphManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Добавляем выпадающие списки команд
        self.setup_team_selection()
        
        # Создаем контейнер для контента 
        self.content_stack = QWidget()
        self.content_layout = QVBoxLayout(self.content_stack)
        self.layout.addWidget(self.content_stack)
        
        # Для статистики
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.content_layout.addWidget(self.stats_display)
        
        # Для графиков
        self.graphs_display = QTextEdit()
        self.graphs_display.setReadOnly(True)
        self.content_layout.addWidget(self.graphs_display)
        self.graphs_display.hide()

    def setup_team_selection(self):
        """Настройка выбора команд (без изменений)"""
        if not self.parent:
            return
            
        layout = QHBoxLayout()
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        
        # Копируем данные из родительского виджета
        for i in range(self.parent.home_combo.count()):
            self.home_combo.addItem(self.parent.home_combo.itemText(i))
            self.away_combo.addItem(self.parent.away_combo.itemText(i))
        
        # Добавляем подписи
        home_layout = QVBoxLayout()
        home_layout.addWidget(QLabel("Домашняя команда:"))
        home_layout.addWidget(self.home_combo)
        
        away_layout = QVBoxLayout()
        away_layout.addWidget(QLabel("Гостевая команда:"))
        away_layout.addWidget(self.away_combo)
        
        layout.addLayout(home_layout)
        layout.addLayout(away_layout)
        self.layout.addLayout(layout)
        
    def on_team_changed(self):
        """Обновляем данные при изменении выбора команд"""
        if hasattr(self, 'home_combo') and hasattr(self, 'away_combo'):
            home_team = self.home_combo.currentText()
            away_team = self.away_combo.currentText()
            self.update_display(home_team, away_team)

    def update_display(self, home_team, away_team):
        """Обновляем отображение статистики/графиков"""
        # Здесь будет логика обновления данных
        pass

    def show_stats(self):
        """Показать статистику и скрыть графики"""
        self.stats_display.show()
        self.graphs_display.hide()  # Скрываем графики

    def show_graphs(self):
        """Показать графики и скрыть статистику"""
        self.stats_display.hide()  # Скрываем статистику
        self.graphs_display.show()