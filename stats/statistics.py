from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel
from PyQt6.QtCore import Qt

class StatisticsManager(QWidget):
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        # Основной layout
        main_layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Статистика команд")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title)
        
        # Двухколоночный layout
        columns = QHBoxLayout()
        
        # Левая колонка (Команда 1)
        left_column = QVBoxLayout()
        left_column.addWidget(QLabel("Команда 1"))
        self.team1_stats = QTextEdit()
        self.team1_stats.setReadOnly(True)
        left_column.addWidget(self.team1_stats)
        
        # Правая колонка (Команда 2)
        right_column = QVBoxLayout()
        right_column.addWidget(QLabel("Команда 2"))
        self.team2_stats = QTextEdit()
        self.team2_stats.setReadOnly(True)
        right_column.addWidget(self.team2_stats)
        
        columns.addLayout(left_column)
        columns.addLayout(right_column)
        main_layout.addLayout(columns)
        
        # Тестовые данные
        self.update_stats("Команда 1", "Команда 2")

    def update_stats(self, team1, team2):
        """Обновляет статистику для обеих команд"""
        self.team1_stats.setPlainText(
            f"Матчи: 100\n"
            f"Победы: 5 (50%)\n"
            f"Форма: 1.75"
        )
        self.team2_stats.setPlainText(
            f"Матчи: 120\n"
            f"Победы: 6 (50%)\n"
            f"Форма: 1.80"
        )
    
    def update_stats(self, home_team, away_team):
        """Обновляет статистику для указанных команд"""
        if not self.predictor or not hasattr(self.predictor, 'df'):
            return
            
        df = self.predictor.df
        
        try:
            # Статистика домашней команды
            home_games = df[df['HomeTeam'] == home_team]
            away_as_home = df[df['AwayTeam'] == home_team]
            
            home_team_stats = (
                f"Статистика для {home_team}:\n"
                f"Всего матчей: {len(home_games) + len(away_as_home)}\n"
                f"Домашние победы: {len(home_games[home_games['FTR'] == 'H'])}\n"
                f"Гостевые победы: {len(away_as_home[away_as_home['FTR'] == 'A'])}\n"
                f"Средняя форма: {home_games['HomeForm'].mean():.2f}"
            )
            
            # Статистика гостевой команды
            away_games = df[df['AwayTeam'] == away_team]
            home_as_away = df[df['HomeTeam'] == away_team]
            
            away_team_stats = (
                f"Статистика для {away_team}:\n"
                f"Всего матчей: {len(away_games) + len(home_as_away)}\n"
                f"Домашние победы: {len(home_as_away[home_as_away['FTR'] == 'H'])}\n"
                f"Гостевые победы: {len(away_games[away_games['FTR'] == 'A'])}\n"
                f"Средняя форма: {away_games['AwayForm'].mean():.2f}"
            )
            
            # Обновляем отображение
            self.home_stats.setPlainText(home_team_stats)
            self.away_stats.setPlainText(away_team_stats)
            
        except Exception as e:
            error_msg = f"Ошибка при загрузке статистики: {str(e)}"
            self.home_stats.setPlainText(error_msg)
            self.away_stats.setPlainText(error_msg)
        
    def show_team_stats(self, home_team: str, away_team: str):
        """Показывает статистику для обеих команд"""
        try:
            df = self.predictor.df
            
            # Статистика домашней команды
            home_stats = self._get_stats(df, home_team, is_home=True)
            
            # Статистика гостевой команды
            away_stats = self._get_stats(df, away_team, is_home=False)
            
            # Формируем итоговый текст
            stats_text = (
                f"=== {home_team} ===\n{home_stats}\n\n"
                f"=== {away_team} ===\n{away_stats}"
            )
            
            self.display.setPlainText(stats_text)
        except Exception as e:
            self.display.setText(f"Ошибка: {str(e)}")

    def _get_stats(self, df, team, is_home):
        """Возвращает статистику для одной команды"""
        if is_home:
            games = df[df['HomeTeam'] == team]
            wins = len(games[games['FTR'] == 'H'])
            form = games['HomeForm'].mean()
        else:
            games = df[df['AwayTeam'] == team]
            wins = len(games[games['FTR'] == 'A'])
            form = games['AwayForm'].mean()
        
        total = len(df[df['HomeTeam'] == team]) + len(df[df['AwayTeam'] == team])
        
        return (
            f"Всего матчей: {total}\n"
            f"Побед: {wins}\n"
            f"Средняя форма: {form:.2f}"
        )
    '''