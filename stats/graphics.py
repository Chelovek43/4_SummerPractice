from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
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

        # Растягиваем GraphsManager на всё пространство
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # График 1: Форма команд
        self.figure1 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas1)
        
        # График 2: Атака/защита
        self.figure2 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas2)

        # График 3: Голы/пропуски
        self.figure3 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas3 = FigureCanvas(self.figure3)
        self.canvas3.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas3) 

        # График 4: Карточки
        self.figure4 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas4 = FigureCanvas(self.figure4)
        self.canvas4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas4)

        # График 5: Удары
        self.figure5 = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas5 = FigureCanvas(self.figure5)
        self.canvas5.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas5)
        
        # Добавляем отступы
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)

    def update_stats(self, team1, team2, period='all'):
        """
        Обновляет все графики для выбранных команд.
        """
        self.team1 = team1
        self.team2 = team2

        # Получаем статистику команд через stats_manager
        stats1 = self.stats_manager.get_team_stats(team1, period=period, opponent=team2 if period == 'h2h' else None)
        stats2 = self.stats_manager.get_team_stats(team2, period=period, opponent=team1 if period == 'h2h' else None)

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

        # График 4: Карточки
        self.figure4.clear()
        ax4 = self.figure4.add_subplot(111)
        self.plot_cards(ax4, stats1, stats2)
        self.canvas4.draw()

        # График 5: Удары
        self.figure5.clear()
        ax5 = self.figure5.add_subplot(111)
        self.plot_shots(ax5, stats1, stats2)
        self.canvas5.draw()
        
    def plot_comparison_bar(
        self,
        ax,
        team1,
        team2,
        stats1,
        stats2,
        labels,
        team1_keys,
        team2_keys,
        ylabel,
        title,
        missing_msg="Данные недоступны"
    ):
        """
        Универсальная функция для построения сравнительного столбчатого графика.

        Args:
            ax: matplotlib.axes.Axes — ось для построения графика.
            team1, team2: str — названия команд.
            stats1, stats2: dict — статистика команд.
            labels: list[str] — подписи для оси X.
            team1_keys: list[str] — ключи для данных team1.
            team2_keys: list[str] — ключи для данных team2.
            ylabel: str — подпись оси Y.
            title: str — заголовок графика.
            missing_msg: str — сообщение при отсутствии данных.
        """
        # Проверка наличия всех нужных ключей
        if not all(k in stats1 for k in team1_keys) or not all(k in stats2 for k in team2_keys):
            ax.text(0.5, 0.5, missing_msg, ha='center', va='center')
            return

        team1_data = [0 if stats1.get(k, 0) is None else stats1.get(k, 0) for k in team1_keys]
        team2_data = [0 if stats2.get(k, 0) is None else stats2.get(k, 0) for k in team2_keys]

        x = range(len(labels))
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], team1_data, width, label=team1)
        bars2 = ax.bar([i + width/2 for i in x], team2_data, width, label=team2)

        # Добавляем запас сверху для подписей (15%)
        all_heights = [bar.get_height() for bar in list(bars1) + list(bars2)]
        max_height = max(all_heights) if all_heights else 1
        ax.set_ylim(top=max_height * 1.15)

        # Добавляем подписи над столбцами
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # смещение по y
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Использование для разных метрик:
    def plot_team_form(self, ax, stats1, stats2):
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Домашняя форма', 'Гостевая форма'],
            team1_keys=['HomeForm', 'AwayForm'],
            team2_keys=['HomeForm', 'AwayForm'],
            ylabel='Форма (средний балл)',
            title=f'Форма команд: {self.team1} vs {self.team2}',
            missing_msg='Данные о форме команд недоступны'
        )

    def plot_attack_defense(self, ax, stats1, stats2):
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Атака хозяев', 'Защита гостей'],
            team1_keys=['HomeAttack', 'AwayDefense'],
            team2_keys=['HomeAttack', 'AwayDefense'],
            ylabel='Показатели',
            title=f'Атака и защита: {self.team1} vs {self.team2}',
            missing_msg='Данные об атаке/защите недоступны'
        )

    def plot_goals_stats(self, ax, stats1, stats2):
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Голы (посл. 3 матча)', 'Пропущено (посл. 3 матча)'],
            team1_keys=['HomeLast3Goals', 'AwayLast3Conceded'],
            team2_keys=['HomeLast3Goals', 'AwayLast3Conceded'],
            ylabel='Средние показатели',
            title=f'Голы и пропуски: {self.team1} vs {self.team2}',
            missing_msg='Данные о голах/пропусках недоступны'
        )

    def plot_cards(self, ax, stats1, stats2):
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Жёлтые', 'Красные'],
            team1_keys=['HY', 'HR'],
            team2_keys=['AY', 'AR'],
            ylabel='Карточки',
            title=f'Карточки: {self.team1} vs {self.team2}',
            missing_msg='Данные о карточках недоступны'
        )

    def plot_shots(self, ax, stats1, stats2):
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Удары', 'В створ'],
            team1_keys=['HS', 'HST'],
            team2_keys=['AS', 'AST'],
            ylabel='Удары',
            title=f'Удары: {self.team1} vs {self.team2}',
            missing_msg='Данные об ударах недоступны'
        )