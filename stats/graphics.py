from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

"""
    Класс для отображения сравнительных графиков двух футбольных команд.
    Включает графики формы, атаки/защиты и голов/пропущенных мячей.
    Использует данные, предоставленные stats_manager.
"""
class GraphsManager(QWidget):
    def __init__(self, stats_manager, parent=None):
        super().__init__(parent)
        self.stats_manager = stats_manager
        self.team1 = ""
        self.team2 = ""
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        # График 1: Форма команд
        self.figure1 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.layout.addWidget(self.canvas1)
        
        # График 2: Атака/защита
        self.figure2 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas2 = FigureCanvas(self.figure2)
        self.layout.addWidget(self.canvas2)

        # График 3: Голы/пропуски
        self.figure3 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas3 = FigureCanvas(self.figure3)
        self.layout.addWidget(self.canvas3)  
        
        # Добавляем отступы
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)
        
    def update_teams(self, team1, team2):
        """Обновление данных команд и графиков"""
        self.team1 = team1
        self.team2 = team2
        self.update_graphs()
    
    def update_graphs(self):
        """Обновление обоих графиков"""
        if not self.team1 or not self.team2:
            return
            
        try:
            stats1 = self.stats_manager.get_team_stats(self.team1)
            stats2 = self.stats_manager.get_team_stats(self.team2)
            
            # График 1: Форма команд
            self.figure1.clear()
            ax1 = self.figure1.add_subplot(111)
            self.plot_team_form(ax1, stats1, stats2)
            self.canvas1.draw()
            
            # График 2: Атака/защита
            self.figure2.clear()
            ax2 = self.figure2.add_subplot(111)
            self.plot_attack_defense(ax2, stats1, stats2)
            self.canvas2.draw()

            # График 3: Голы/пропуски
            self.figure3.clear()
            ax3 = self.figure3.add_subplot(111)
            self.plot_goals_stats(ax3, stats1, stats2)
            self.canvas3.draw()
            
        except Exception as e:
            print(f"Ошибка при обновлении графиков: {e}")
    
    def plot_team_form(self, ax, stats1, stats2):
        """График формы команд"""
        if 'HomeForm' not in stats1 or 'AwayForm' not in stats1:
            ax.text(0.5, 0.5, 'Данные о форме команд недоступны', 
                   ha='center', va='center')
            return
                
        labels = ['Домашняя форма', 'Гостевая форма']
        team1_data = [stats1.get('HomeForm', 0), stats1.get('AwayForm', 0)]
        team2_data = [stats2.get('HomeForm', 0), stats2.get('AwayForm', 0)]
        
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width/2 for i in x], team1_data, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_data, width, label=self.team2)
        
        ax.set_ylabel('Форма (средний балл)')
        ax.set_title(f'Форма команд: {self.team1} vs {self.team2}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    def plot_attack_defense(self, ax, stats1, stats2):
        """График атаки и защиты"""
        if 'HomeAttack' not in stats1 or 'AwayDefense' not in stats2:
            ax.text(0.5, 0.5, 'Данные об атаке/защите недоступны', 
                   ha='center', va='center')
            return
            
        categories = ['Атака хозяев', 'Защита гостей']
        team1_values = [stats1.get('HomeAttack', 0), stats2.get('AwayDefense', 0)]
        team2_values = [stats2.get('HomeAttack', 0), stats1.get('AwayDefense', 0)]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], team1_values, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_values, width, label=self.team2)
        
        ax.set_ylabel('Показатели')
        ax.set_title(f'Атака и защита: {self.team1} vs {self.team2}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    def plot_goals_stats(self, ax, stats1, stats2):
        """График голов и пропущенных мячей"""
        if 'HomeLast3Goals' not in stats1 or 'AwayLast3Conceded' not in stats1:
            ax.text(0.5, 0.5, 'Данные о голаx/пропусках недоступны', 
                ha='center', va='center')
            return
            
        categories = ['Голы (посл. 3 матча)', 'Пропущено (посл. 3 матча)']
        team1_values = [stats1.get('HomeLast3Goals', 0), stats1.get('AwayLast3Conceded', 0)]
        team2_values = [stats2.get('HomeLast3Goals', 0), stats2.get('AwayLast3Conceded', 0)]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], team1_values, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_values, width, label=self.team2)
        
        ax.set_ylabel('Средние показатели')
        ax.set_title(f'Голы и пропуски: {self.team1} vs {self.team2}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)   

    
    