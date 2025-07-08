from PyQt6.QtWidgets import QTextEdit
import pandas as pd

class StatisticsManager:
    def __init__(self, text_edit: QTextEdit, predictor):
        self.display = text_edit
        self.predictor = predictor


        '''
        def show_team_stats(self, team: str):
        """Показывает базовую статистику команды"""
        try:
            # Используем данные, уже загруженные в predictor
            df = self.predictor.df
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]
            
            stats = f"""
            Статистика для {team}:
            Всего матчей: {len(home_games) + len(away_games)}
            Домашние победы: {len(home_games[home_games['FTR'] == 'H'])}
            Гостевые победы: {len(away_games[away_games['FTR'] == 'A'])}
            Средняя форма: {home_games['HomeForm'].mean():.2f}
            """
            self.display.setPlainText(stats)
        except Exception as e:
            self.display.setText(f"Ошибка: {str(e)}")
        '''
    