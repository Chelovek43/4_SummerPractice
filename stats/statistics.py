import pandas as pd

class StatisticsManager:
    def __init__(self, df=None):
        """
        Args:
            df (pd.DataFrame): готовый DataFrame с данными
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
        print(self.df.head())
        print(self.df.columns.tolist())
        """Возвращает усредненные значения из подготовленного DataFrame"""
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        # Фильтрация данных
        df_filtered = self.filter_by_period(team_name, period, opponent)
        
        home_matches = df_filtered[df_filtered['HomeTeam'] == team_name]
        away_matches = df_filtered[df_filtered['AwayTeam'] == team_name]
        
        # Расчет показателей без использования HomeWin/AwayWin/Draw
        home_wins = (home_matches['FTR'] == 'H').sum()
        away_wins = (away_matches['FTR'] == 'A').sum()
        home_draws = (home_matches['FTR'] == 'D').sum()
        away_draws = (away_matches['FTR'] == 'D').sum()
        
        total_matches = len(home_matches) + len(away_matches)
        wins = home_wins + away_wins
        draws = home_draws + away_draws
        
        
        return {
            'total_matches': total_matches,
            'wins': wins,
            'draws': draws,
            'win_rate': ((wins + 0.5 * draws) / total_matches * 100) if total_matches > 0 else 0,
            
            # Берем средние значения из подготовленных столбцов
            'HomeForm': home_matches['HomeForm'].mean(),
            'AwayForm': away_matches['AwayForm'].mean(),
            'HomeAttack': home_matches['HomeAttack'].mean(),
            'AwayDefense': away_matches['AwayDefense'].mean(),
            'HomeLast3Goals': home_matches['HomeLast3Goals'].mean(),
            'AwayLast3Conceded': away_matches['AwayLast3Conceded'].mean(),
        }


    def filter_by_period(self, team_name, period, opponent=None):
        """Фильтрует данные по периоду"""
        if period == 'h2h' and opponent:
            return self.df[
                ((self.df['HomeTeam'] == team_name) & (self.df['AwayTeam'] == opponent)) |
                ((self.df['HomeTeam'] == opponent) & (self.df['AwayTeam'] == team_name))
            ]
        elif period.isdigit():
            n = int(period)
            home = self.df[self.df['HomeTeam'] == team_name].tail(n)
            away = self.df[self.df['AwayTeam'] == team_name].tail(n)
            return pd.concat([home, away]).sort_index().tail(n)
        return self.df

    def get_head_to_head(self, team1, team2):
        """Статистика личных встреч"""
        h2h = self.df[
            ((self.df['HomeTeam'] == team1) & (self.df['AwayTeam'] == team2)) |
            ((self.df['HomeTeam'] == team2) & (self.df['AwayTeam'] == team1))
        ]
        
        if h2h.empty:
            return {}
        
        team1_wins = len(h2h[((h2h['HomeTeam'] == team1) & (h2h['FTR'] == 'H')) | 
                        ((h2h['AwayTeam'] == team1) & (h2h['FTR'] == 'A'))])
        
        return {
            'HeadToHeadMatches': len(h2h),
            'HeadToHeadWinRate': (team1_wins / len(h2h)) * 100,
            'HeadToHeadAvgGoals': h2h.apply(
                lambda x: x['FTHG'] if x['HomeTeam'] == team1 else x['FTAG'], axis=1).mean()
        }