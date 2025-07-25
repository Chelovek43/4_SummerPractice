from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout,QFrame
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

    
    """
    def plot_attack_defense(self, stats1, stats2):
        pass
        
    def plot_goals_conceded(self, stats1, stats2):
        pass
        
    def plot_head_to_head(self, stats1, stats2):
        pass
    """



'''
class GraphsManager(QWidget):
    def __init__(self, stats_manager, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.stats_manager = stats_manager
        self.current_period = 'all'
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Выбор типа графика
        self.setup_graph_selection()
        
        # Контейнер для графиков
        self.setup_graphs_container()
        
        # Инициализация графиков
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.graphs_container.addWidget(self.canvas)

        # Временный лог для проверки
        print("Все дочерние виджеты:", self.findChildren(QWidget))
        for widget in self.findChildren(QWidget):
            print(f"Виджет: {widget}, Геометрия: {widget.geometry()}")
        
        # Или выделите красной рамкой проблемный элемент
        debug_frame = QFrame(self)
        debug_frame.setStyleSheet("border: 2px solid red;")
        debug_frame.setGeometry(0, 0, self.width(), 30)  # Подстройте размеры
        debug_frame.lower()  # Отправить на задний план
        
    def setup_graph_selection(self):
        """Настройка выбора типа графика"""
        graph_layout = QHBoxLayout()
        
        graph_label = QLabel("Тип графика:")
        self.graph_combo = QComboBox()
        self.graph_combo.addItem("Форма команд", "form")
        self.graph_combo.addItem("Атака/защита", "attack_defense")
        self.graph_combo.addItem("Голы/пропуски", "goals")
        self.graph_combo.addItem("Личные встречи", "h2h")
        self.graph_combo.currentIndexChanged.connect(self.update_graphs)
        
        graph_layout.addWidget(graph_label)
        graph_layout.addWidget(self.graph_combo)
        graph_layout.addStretch()
        
        self.layout.addLayout(graph_layout)
    
    def setup_graphs_container(self):
        """Настройка контейнера для графиков"""
        self.graphs_container = QVBoxLayout()
        self.graphs_container.setContentsMargins(0, 0, 0, 0)
        self.layout.addLayout(self.graphs_container)
    
    def update_teams(self, team1, team2):
        """Обновление команд для графиков"""
        self.team1 = team1
        self.team2 = team2
        self.update_graphs()
    
    def update_period(self, period):
        """Обновление периода для графиков"""
        self.current_period = period
        self.update_graphs()
    
    def update_graphs(self):
        """Обновление графиков на основе выбранных параметров"""
        if not hasattr(self, 'team1') or not hasattr(self, 'team2'):
            return
            
        # Получаем статистику для команд
        stats1 = self.stats_manager.get_team_stats(
            self.team1, 
            period=str(self.current_period),
            opponent=self.team2 if self.current_period == 'h2h' else None
        )
        
        stats2 = self.stats_manager.get_team_stats(
            self.team2,
            period=str(self.current_period),
            opponent=self.team1 if self.current_period == 'h2h' else None
        )
        
        # Очищаем предыдущий график
        self.figure.clear()
        
        # Создаем новый график в зависимости от выбранного типа
        graph_type = self.graph_combo.currentData()
        
        if graph_type == "form":
            self.plot_team_form(stats1, stats2)
        elif graph_type == "attack_defense":
            self.plot_attack_defense(stats1, stats2)
        elif graph_type == "goals":
            self.plot_goals_conceded(stats1, stats2)
        elif graph_type == "h2h" and 'HeadToHeadMatches' in stats1:
            self.plot_head_to_head(stats1, stats2)
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Данные для графиков недоступны', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        self.canvas.draw()
    
    def plot_team_form(self, stats1, stats2):
        """График формы команд"""
        ax = self.figure.add_subplot(111)
        
        labels = ['Домашняя форма', 'Гостевая форма']
        team1_values = [stats1.get('HomeForm', 0), stats1.get('AwayForm', 0)]
        team2_values = [stats2.get('HomeForm', 0), stats2.get('AwayForm', 0)]
        
        x = range(len(labels))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], team1_values, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_values, width, label=self.team2)
        
        ax.set_ylabel('Форма (средний балл)')
        ax.set_title('Сравнение формы команд')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    def plot_attack_defense(self, stats1, stats2):
        """График атаки и защиты"""
        ax = self.figure.add_subplot(111)
        
        categories = ['Атака хозяев', 'Защита гостей']
        team1_values = [stats1.get('HomeAttack', 0), stats2.get('AwayDefense', 0)]
        team2_values = [stats2.get('HomeAttack', 0), stats1.get('AwayDefense', 0)]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], team1_values, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_values, width, label=self.team2)
        
        ax.set_ylabel('Показатели')
        ax.set_title('Сравнение атаки и защиты')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    def plot_goals_conceded(self, stats1, stats2):
        """График голов и пропущенных мячей"""
        ax = self.figure.add_subplot(111)
        
        categories = ['Голы (посл. 3 матча)', 'Пропущено (посл. 3 матча)']
        team1_values = [stats1.get('HomeLast3Goals', 0), stats1.get('AwayLast3Conceded', 0)]
        team2_values = [stats2.get('HomeLast3Goals', 0), stats2.get('AwayLast3Conceded', 0)]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], team1_values, width, label=self.team1)
        ax.bar([i + width/2 for i in x], team2_values, width, label=self.team2)
        
        ax.set_ylabel('Среднее количество')
        ax.set_title('Голы и пропущенные мячи')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    def plot_head_to_head(self, stats1, stats2):
        """График личных встреч"""
        ax = self.figure.add_subplot(111)
        
        labels = [self.team1, self.team2]
        win_rates = [stats1.get('HeadToHeadWinRate', 0), 
                   100 - stats1.get('HeadToHeadWinRate', 0)]
        avg_goals = [stats1.get('HeadToHeadAvgGoals', 0), 
                    stats2.get('HeadToHeadAvgGoals', 0)]
        
        x = range(len(labels))
        width = 0.35
        
        # График процента побед
        ax.bar([i - width/2 for i in x], win_rates, width, label='Процент побед')
        
        # График средних голов
        ax2 = ax.twinx()
        ax2.bar([i + width/2 for i in x], avg_goals, width, color='orange', label='Средние голы')
        
        ax.set_ylabel('Процент побед (%)')
        ax2.set_ylabel('Средние голы')
        ax.set_title('Личные встречи команд')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Объединяем легенды
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
        
        ax.grid(True, linestyle='--', alpha=0.6)

'''