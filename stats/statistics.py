import pandas as pd

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel
from PyQt6.QtCore import Qt

from core.predicrtor import FootballMatchPredictor

class StatisticsManager:
    def __init__(self, df=None):
        """
        Args:
            df (pd.DataFrame): Опционально - готовый DataFrame с данными
        """
        self.df = df
        
    def load_data(self, data_path):
        """Загрузка данных из CSV файла"""
        self.df = pd.read_csv(data_path)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Базовая подготовка данных"""
        if self.df is None:
            return
            
        # Создаем числовые колонки для результатов
        self.df['HomeWin'] = (self.df['FTR'] == 'H').astype(int)
        self.df['AwayWin'] = (self.df['FTR'] == 'A').astype(int)
        self.df['Draw'] = (self.df['FTR'] == 'D').astype(int)
        
        # Форма команды (последние 5 матчей)
        self.df['HomeForm'] = self.df.groupby('HomeTeam')['HomeWin'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        self.df['AwayForm'] = self.df.groupby('AwayTeam')['AwayWin'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())

    def get_team_stats(self, team_name, period='all', opponent=None):
        print(f"\nGetting stats for {team_name}, period: {period}, opponent: {opponent}")
    
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        df_filtered = self.df.copy()
        
        # Фильтрация по периоду
        if period == 'h2h' and opponent:
            print("Filtering head-to-head matches")
            df_filtered = df_filtered[
                ((df_filtered['HomeTeam'] == team_name) & (df_filtered['AwayTeam'] == opponent)) |
                ((df_filtered['HomeTeam'] == opponent) & (df_filtered['AwayTeam'] == team_name))
            ]
        elif period.isdigit():
            print(f"Filtering last {period} matches")
            n = int(period)
            home_matches = df_filtered[df_filtered['HomeTeam'] == team_name].tail(n)
            away_matches = df_filtered[df_filtered['AwayTeam'] == team_name].tail(n)
            df_filtered = pd.concat([home_matches, away_matches]).sort_index().tail(n)
        
        print(f"Found {len(df_filtered)} matches after filtering")
        
        # Расчет статистики
        home_matches = df_filtered[df_filtered['HomeTeam'] == team_name]
        away_matches = df_filtered[df_filtered['AwayTeam'] == team_name]
        
        print(f"Home matches: {len(home_matches)}, Away matches: {len(away_matches)}")
        
        total_matches = len(home_matches) + len(away_matches)
        wins = home_matches['HomeWin'].sum() + away_matches['AwayWin'].sum()
        draws = home_matches['Draw'].sum() + away_matches['Draw'].sum()
        
        return {
            'matches': total_matches,
            'wins': wins,
            'win_rate': round((wins + 0.5 * draws) / total_matches * 100, 2) if total_matches > 0 else 0,
            'form': round((
                home_matches['HomeForm'].mean() if not home_matches.empty else 0 + 
                away_matches['AwayForm'].mean() if not away_matches.empty else 0
            ) / 2, 2)
        }