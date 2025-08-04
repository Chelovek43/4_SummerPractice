from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QComboBox, QHBoxLayout, QScrollArea, QSizePolicy, QMessageBox
from PyQt6.QtCore import Qt
from stats.graphics import GraphsManager

# TODO: Класс StatsGraphManager отвечает и за отображение статистики, и за графики, и за выбор команд
class StatsGraphManager(QWidget):
    def __init__(self, stats_manager, parent=None, show_team_select=True):
        """
        Конструктор класса для виджета статистики и графиков.

        stats_manager: Экземпляр менеджера статистики, предоставляющий данные о командах.
        parent: Родительский виджет (по умолчанию None).
        show_team_select: Флаг, определяющий, отображать ли выбор команд (по умолчанию True).
        """
        super().__init__(parent)  
        self.parent = parent  
        self.stats_manager = stats_manager 
        self.show_team_select = show_team_select  # Флаг для отображения выбора команд
        self.current_period = 'all'  # Текущий выбранный период статистики (по умолчанию — вся статистика)
        self.init_ui()  

    def init_ui(self):
        """
        Инициализация элементов пользовательского интерфейса.
        Создаёт основной layout и добавляет необходимые компоненты:
        выбор команд, выбор периода, отображение статистики и графиков.
        """
        self.layout = QVBoxLayout(self)  

        # Добавляем выбор команд только если это предусмотрено флагом show_team_selection
        if self.show_team_select:
            self.setup_team_selection()  # Настройка выпадающих списков для выбора команд

        self.setup_period_selection()  # Настройка выбора периода статистики (например, последние 5 матчей)
        self.setup_stats_display()     # Настройка области отображения статистики команд
        self.setup_graphs_display()    # Настройка области отображения графиков
        self.show_stats()              

    def setup_period_selection(self):
        """Настройка выбора периода статистики"""
        period_layout = QHBoxLayout()
        period_label = QLabel("Период статистики:")
        self.period_combo = QComboBox()
        # Добавляем варианты выбора периода в выпадающий список
        self.period_combo.addItem("Вся статистика", "all")
        self.period_combo.addItem("Последние 5 матчей", "5")
        self.period_combo.addItem("Последние 10 матчей", "10")
        self.period_combo.addItem("Весь сезон (38 матчей)", "38")
        self.period_combo.addItem("Только между этими командами", "h2h")
        # Подключаем обработчик изменения выбранного периода
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        
        period_layout.addWidget(period_label)
        period_layout.addWidget(self.period_combo)
        # Добавляем растягивающееся пространство для выравнивания
        period_layout.addStretch()
        
        # Добавляем слой выбора периода в основной макет
        self.layout.addLayout(period_layout)

    def on_period_changed(self):
        """
        Обработчик события изменения выбранного периода статистики в выпадающем списке.

        При изменении значения в self.period_combo:
        - Сохраняет новое значение периода в self.current_period.
        - Обновляет отображаемую статистику с учётом выбранного периода.
        """
        self.current_period = self.period_combo.currentData()  
        self.refresh_stats()  # Перерисовываем статистику для новых параметров

    def setup_stats_display(self):
        """
        Настраивает область отображения статистики команд.

        Создаёт заголовок, две колонки для статистики каждой команды и размещает их в горизонтальном layout.
        """

        self.stats_title = QLabel("Статистика команд")
        self.stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.stats_title)
        
        # Основной контейнер 
        stats_widget = QWidget()
        stats_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stats_container = QHBoxLayout(stats_widget)
        self.stats_container.setContentsMargins(0, 0, 0, 0) # Без внутренних отступов
        self.stats_container.setSpacing(10)
        
        # Колонка 1
        team1_widget = QWidget()
        team1_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.team1_column = QVBoxLayout(team1_widget)
        self.team1_column.setContentsMargins(0, 0, 0, 0)
        
        self.team1_label = QLabel("Команда 1")
        self.team1_label.setStyleSheet("font-weight: bold;")
        
        self.team1_stats = QTextEdit()
        self.team1_stats.setReadOnly(True)
        self.team1_stats.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.team1_stats.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.team1_stats.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.team1_column.addWidget(self.team1_label)
        self.team1_column.addWidget(self.team1_stats, stretch=1)  
        
        # Колонка 2 (аналогично)
        team2_widget = QWidget()
        team2_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.team2_column = QVBoxLayout(team2_widget)
        self.team2_column.setContentsMargins(0, 0, 0, 0)
        
        self.team2_label = QLabel("Команда 2")
        self.team2_label.setStyleSheet("font-weight: bold;")
        
        self.team2_stats = QTextEdit()
        self.team2_stats.setReadOnly(True)
        self.team2_stats.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.team2_stats.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.team2_stats.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.team2_column.addWidget(self.team2_label)
        self.team2_column.addWidget(self.team2_stats, stretch=1)
        
        # Добавляем колонки
        self.stats_container.addWidget(team1_widget, stretch=1)
        self.stats_container.addWidget(team2_widget, stretch=1)
        
        # Добавляем в главный layout
        self.layout.addWidget(stats_widget, stretch=1)  


    def setup_graphs_display(self):
        """Настройка отображения графиков с прокруткой"""
        # Создаем ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Позволяет содержимому изменять размер вместе с окном
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Создаем контейнер для графиков
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)  # Отступы
        
        # Добавляем GraphsManager с фиксированными размерами
        self.graphs_manager = GraphsManager(self.stats_manager, self)
        self.graphs_manager.setMinimumSize(400,300)  # Минимальный размер
        layout.addWidget(self.graphs_manager)
        
        scroll.setWidget(container)
        self.layout.addWidget(scroll)  # Добавляем в основной layout
        
        self.graphs_manager.hide()

    def check_same_teams(self, home_team, away_team):
        """
        Проверяет, выбраны ли одинаковые команды.
        Если да — показывает QMessageBox и возвращает True.
        Если нет — возвращает False.
        """
        if home_team == away_team:
            QMessageBox.warning(self, "Ошибка", "Команды не должны быть одинаковыми!")
            return True
        return False
    
    def display_stats(self, team1, team2, period):
        """Получает, форматирует и отображает статистику для двух команд."""
        stats1 = self.stats_manager.get_team_stats(team1, period=str(period),
                                                opponent=team2 if period == 'h2h' else None)
        stats2 = self.stats_manager.get_team_stats(team2, period=str(period),
                                                opponent=team1 if period == 'h2h' else None)
        self.team1_stats.setPlainText(self.format_stats(stats1))
        self.team2_stats.setPlainText(self.format_stats(stats2))
        
    def update_stats(self, team1, team2):
        """Обновление статистики и графиков"""
        if self.check_same_teams(team1, team2):
            return
    
        self.team1_name = team1
        self.team2_name = team2
        
        # Обновляем текстовую статистику
        self.display_stats(team1, team2, self.current_period)
        
        # Если графики видны - обновляем их
        if self.graphs_manager.isVisible():
            self.graphs_manager.update_teams(team1, team2)

    def refresh_stats(self):
        """
        Обновление отображения статистики для выбранных команд и периода.

        Использует текущие значения self.team1_name и self.team2_name, а также self.current_period.
        Если выбран режим 'h2h' (личные встречи), то статистика собирается только для матчей между этими командами.
        В противном случае — по общему периоду.
        """
        # Проверяем, что имена обеих команд заданы (иначе обновлять нечего)
        if not hasattr(self, 'team1_name') or not hasattr(self, 'team2_name'):
            return
        
        if self.check_same_teams(self.team1_name, self.team2_name):
            return
        
        # Отладка
        print(f"\nRefreshing stats with period: {self.current_period}")
        
        self.display_stats(self.team1_name, self.team2_name, self.current_period)

    def format_stats(self, stats):
        """Форматирует словарь статистики команды в удобочитаемый текст для отображения в UI"""

        # Если статистика отсутствует (None или пустой словарь)
        if not stats:
            return "Статистика недоступна"
        
        try:
            text = (
                f" Основная статистика:\n"
                f"- Всего матчей: {int(stats['total_matches'])}\n"
                f"- Победы: {int(stats['wins'])}\n"
                f"- Ничьи: {int(stats['draws'])}\n"
                f"- Процент побед: {stats['win_rate']:.1f}%\n\n"
                
                f" Форма команды:\n"
                f"- Домашняя форма: {stats['HomeForm']:.2f}\n"
                f"- Гостевая форма: {stats['AwayForm']:.2f}\n\n"
                
                f" Атака/защита:\n"
                f"- Сила атаки (дома): {stats['HomeAttack']:.2f}\n"
                f"- Надежность защиты (в гостях): {stats['AwayDefense']:.2f}\n"
                f"- Голов за последние 3 матча: {stats['HomeLast3Goals']:.1f}\n"
                f"- Пропущено за последние 3 матча: {stats['AwayLast3Conceded']:.1f}"
            )
            
            # Если в статистике присутствуют данные о личных встречах — добавляем их в текст
            if 'HeadToHeadWinRate' in stats:
                text += (
                    f"\n\n Личные встречи:\n"
                    f"- Всего матчей: {int(stats['HeadToHeadMatches'])}\n"
                    f"- Процент побед: {stats['HeadToHeadWinRate']:.1f}%\n"
                    f"- Средние голы: {stats['HeadToHeadAvgGoals']:.1f}"
                )
                
            return text
            
        except Exception as e:
            print(f"Ошибка форматирования: {e}")
            return "Ошибка при формировании статистики"
        
    def show_stats(self):
        """
    Показывает элементы статистики команд и скрывает графики.

    Используется для переключения отображения между статистикой и графиками.
    Делает видимыми заголовок, названия команд и текстовые поля статистики.
    Если менеджер графиков существует, скрывает его.
    """
        self.stats_title.show()
        self.team1_label.show()
        self.team1_stats.show()
        self.team2_label.show()
        self.team2_stats.show()
        
        # Скрываем менеджер графиков, если он есть
        if hasattr(self, 'graphs_manager'):
            self.graphs_manager.hide()

    def show_graphs(self):
        """
    Показывает графики и скрывает элементы статистики команд.

    Используется для переключения отображения между графиками и статистикой.
    Делает невидимыми заголовок, названия команд и текстовые поля статистики.
    Если менеджер графиков существует, обновляет его для выбранных команд и показывает.
    """
        
        self.stats_title.hide()
        self.team1_label.hide()
        self.team1_stats.hide()
        self.team2_label.hide()
        self.team2_stats.hide()
        
        if hasattr(self, 'graphs_manager'):
            # Перед показом графиков обновляем их для выбранных команд
            if hasattr(self, 'team1_name') and hasattr(self, 'team2_name'):
                self.graphs_manager.update_teams(self.team1_name, self.team2_name)
            self.graphs_manager.show()

    def setup_team_selection(self):
        """
    Настройка выпадающих списков для выбора домашних и гостевых команд с синхронизацией с родительским виджетом.

    - Проверяет наличие родителя и необходимых атрибутов.
    - Копирует команды из родительских комбобоксов.
    - Устанавливает текущие выбранные значения.
    - Создаёт и размещает подписи и комбобоксы в layout.
    - Подключает сигналы изменения выбора к обработчику родителя для синхронизации.
    """
        
        # Проверяем, что у родителя есть необходимые комбобоксы для выбора команд
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
