import logging
import itertools

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class GraphsManager(QWidget):
    """
    Класс для управления и отображения различных статистических графиков по двум командам
    Использует matplotlib для построения графиков 
    Позволяет динамически обновлять графики на основе выбранных команд и периода
    """

    GRAPH_CONFIGS = (
        {
            "plot_func": "plot_team_form",
            "title": "Форма команд",
        },
        {
            "plot_func": "plot_attack_defense",
            "title": "Атака/защита",
        },
        {
            "plot_func": "plot_goals_stats",
            "title": "Голы/пропуски",
        },
        {
            "plot_func": "plot_cards",
            "title": "Карточки",
        },
        {
            "plot_func": "plot_shots",
            "title": "Удары",
        },
    )

    def __init__(self, stats_manager, parent=None):
        """
        Инициализация менеджера графиков.

        stats_manager: Объект для получения статистики команд.
        parent: Родительский виджет (по умолчанию None).
        """
        super().__init__(parent)
        self.stats_manager = stats_manager
        self.team1 = ""
        self.team2 = ""
        self.figures = []
        self.canvases = []
        self.init_ui()

    def init_ui(self):
        """
        Создаёт и настраивает графические элементы интерфейса (фигуры и канвасы) для всех графиков.
        """
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Создание фигур и канвасов для каждого графика
        for _ in self.GRAPH_CONFIGS:
            fig = Figure(figsize=(8, 3), tight_layout=True)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.layout.addWidget(canvas)
            self.figures.append(fig)
            self.canvases.append(canvas)

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)

    def update_stats(self, team1, team2, period='all'):
        """
        Обновляет все графики для выбранных команд и периода.

        team1: Название первой команды.
        team2: Название второй команды.
        period: Период статистики ('all', 'h2h' и т.д.).
        """
        self.team1 = team1
        self.team2 = team2

        try:
            # Получение статистики для обеих команд
            stats1 = self.stats_manager.get_team_stats(
                team1, period=period, opponent=team2 if period == 'h2h' else None
            )
            stats2 = self.stats_manager.get_team_stats(
                team2, period=period, opponent=team1 if period == 'h2h' else None
            )
        except (KeyError, ValueError) as e:
            logging.error(f"Ошибка получения статистики: {e}")
            stats1, stats2 = None, None

        if not stats1 or not stats2:
            # Если статистика недоступна — показать ошибку на всех графиках
            self.show_error_on_all("Ошибка загрузки статистики")
            return

        # Универсальное обновление графиков по конфигурации
        for idx, config in enumerate(self.GRAPH_CONFIGS):
            fig = self.figures[idx]
            canvas = self.canvases[idx]
            fig.clear()
            ax = fig.add_subplot(111)
            plot_func = getattr(self, config["plot_func"])
            plot_func(ax, stats1, stats2)
            canvas.draw()

    def show_error_on_all(self, msg):
        """
        Отображает сообщение об ошибке на всех графиках.

        msg: Текст сообщения об ошибке.
        """
        for fig, canvas in zip(self.figures, self.canvases):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw()

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
        Универсальный метод построения столбчатых сравнительных графиков для двух команд.

        ax: Объект оси matplotlib для построения графика.
        team1: Название первой команды.
        team2: Название второй команды.
        stats1: Статистика первой команды (dict).
        stats2: Статистика второй команды (dict).
        labels: Подписи для оси X.
        team1_keys: Ключи статистики для первой команды.
        team2_keys: Ключи статистики для второй команды.
        ylabel: Подпись оси Y.
        title: Заголовок графика.
        missing_msg: Сообщение при отсутствии данных.
        """
        # Проверка наличия всех необходимых данных
        if not all(k in stats1 for k in team1_keys) or not all(k in stats2 for k in team2_keys):
            ax.text(0.5, 0.5, missing_msg, ha='center', va='center')
            return

        # Получение данных для построения
        team1_data = [0 if stats1.get(k, 0) is None else stats1.get(k, 0) for k in team1_keys]
        team2_data = [0 if stats2.get(k, 0) is None else stats2.get(k, 0) for k in team2_keys]

        x = range(len(labels))
        width = 0.35
        # Построение столбцов для обеих команд
        bars1 = ax.bar([i - width/2 for i in x], team1_data, width, label=team1)
        bars2 = ax.bar([i + width/2 for i in x], team2_data, width, label=team2)

        # Автоматическое масштабирование оси Y
        all_heights = [bar.get_height() for bar in itertools.chain(bars1, bars2)]
        max_height = max(all_heights) if all_heights else 1
        ax.set_ylim(top=max_height * 1.15)

        # Подписи над столбцами
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
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

    def plot_team_form(self, ax, stats1, stats2):
        """
        Строит график формы команд (домашняя и гостевая форма).

        ax: Ось matplotlib.
        stats1: Статистика первой команды.
        stats2: Статистика второй команды.
        """
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
        """
        Строит график сравнения атаки и защиты команд
        """
        self.plot_comparison_bar(
            ax,
            self.team1, self.team2,
            stats1, stats2,
            labels=['Атака', 'Защита'],
            team1_keys=['HomeAttack', 'AwayDefense'],
            team2_keys=['HomeAttack', 'AwayDefense'],
            ylabel='Показатели',
            title=f'Атака и защита: {self.team1} vs {self.team2}',
            missing_msg='Данные об атаке/защите недоступны'
        )

    def plot_goals_stats(self, ax, stats1, stats2):
        """
        Строит график голов и пропущенных мячей за последние 3 матча.
        """
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
        """
        Строит график по жёлтым и красным карточкам.
        """
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
        """
        Строит график по ударам и ударам в створ.
        """
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