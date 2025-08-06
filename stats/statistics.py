import pandas as pd

class StatisticsManager:
    def __init__(self, df=None):
        """
            df (pd.DataFrame): готовый DataFrame с данными
        """
        self.df = df
        
    def load_data(self, data_path):
        """Загрузка данных из CSV файла"""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Выполняет базовую подготовку данных:
        - Добавляет бинарные колонки для результата матча (победа хозяев, победа гостей, ничья).
        - Вычисляет форму команды (доля побед за последние 5 матчей) для хозяев и гостей.
        """
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

        # Примерные расчёты для attack/defense, goals, карточек, ударов
        self.df['HomeAttack'] = self.df['FTHG']
        self.df['AwayDefense'] = self.df['FTAG']
        self.df['HomeLast3Goals'] = self.df.groupby('HomeTeam')['FTHG'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        self.df['AwayLast3Conceded'] = self.df.groupby('AwayTeam')['FTAG'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    def get_team_stats(self, team_name, period='all', opponent=None):
        """
        Возвращает усреднённые статистические показатели для заданной команды за указанный период.

        team_name: Название команды.
        period: Период анализа ('all', число последних матчей или 'h2h' для личных встреч).
        opponent: Имя соперника (используется для 'h2h').
        return: Словарь с основными статистическими показателями.
        """
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        # Фильтрация данных
        df_filtered = self.filter_by_period(team_name, period, opponent)

        # Разделяем матчи на домашние и гостевые для выбранной команды
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

        # Карточки и удары
        home_yellow = home_matches['HY'].mean() if 'HY' in home_matches and not home_matches.empty else 0
        away_yellow = away_matches['AY'].mean() if 'AY' in away_matches and not away_matches.empty else 0
        home_red = home_matches['HR'].mean() if 'HR' in home_matches and not home_matches.empty else 0
        away_red = away_matches['AR'].mean() if 'AR' in away_matches and not away_matches.empty else 0

        home_shots = home_matches['HS'].mean() if 'HS' in home_matches and not home_matches.empty else 0
        away_shots = away_matches['AS'].mean() if 'AS' in away_matches and not away_matches.empty else 0
        home_shots_on_target = home_matches['HST'].mean() if 'HST' in home_matches and not home_matches.empty else 0
        away_shots_on_target = away_matches['AST'].mean() if 'AST' in away_matches and not away_matches.empty else 0

        # Head-to-head winrate (если нужен для графика)
        h2h_stats = self.get_head_to_head(team_name, opponent) if opponent else {}

        # Гарантируем, что все значения числовые (0 если None)
        def safe_mean(val):
            return 0 if pd.isnull(val) else val

        return {
            'total_matches': total_matches,
            'wins': wins,
            'draws': draws,
            'win_rate': ((wins + 0.5 * draws) / total_matches * 100) if total_matches > 0 else 0,
            'HomeForm': safe_mean(home_matches['HomeForm'].mean()) if not home_matches.empty else 0,
            'AwayForm': safe_mean(away_matches['AwayForm'].mean()) if not away_matches.empty else 0,
            'HomeAttack': safe_mean(home_matches['HomeAttack'].mean()) if not home_matches.empty else 0,
            'AwayDefense': safe_mean(away_matches['AwayDefense'].mean()) if not away_matches.empty else 0,
            'HomeLast3Goals': safe_mean(home_matches['HomeLast3Goals'].mean()) if not home_matches.empty else 0,
            'AwayLast3Conceded': safe_mean(away_matches['AwayLast3Conceded'].mean()) if not away_matches.empty else 0,
            # Карточки
            'HY': safe_mean(home_yellow),
            'AY': safe_mean(away_yellow),
            'HR': safe_mean(home_red),
            'AR': safe_mean(away_red),
            # Удары
            'HS': safe_mean(home_shots),
            'AS': safe_mean(away_shots),
            'HST': safe_mean(home_shots_on_target),
            'AST': safe_mean(away_shots_on_target),
        }

    def filter_by_period(self, team_name, period, opponent=None):
        """
        Фильтрует DataFrame по заданному периоду:
        - 'h2h' и opponent: только личные встречи между командами.
        - число: последние N матчей команды (дом+гости).
        - 'all': все матчи.

        team_name: Название команды.
        period: Период ('all', число, 'h2h').
        opponent: Имя соперника (для 'h2h').
        return: Отфильтрованный DataFrame.
        """

        # Личные встречи между двумя командами
        if period == 'h2h' and opponent:
            return self.df[
                ((self.df['HomeTeam'] == team_name) & (self.df['AwayTeam'] == opponent)) |
                ((self.df['HomeTeam'] == opponent) & (self.df['AwayTeam'] == team_name))
            ]
        # Последние N матчей (домашние и гостевые)
        elif isinstance(period, str) and period.isdigit():
            n = int(period)
            home = self.df[self.df['HomeTeam'] == team_name].tail(n)
            away = self.df[self.df['AwayTeam'] == team_name].tail(n)
            return pd.concat([home, away]).sort_index().tail(n)
        return self.df

    def get_head_to_head(self, team1, team2):
        """
        Возвращает статистику личных встреч между двумя командами.

        team1: Название первой команды.
        team2: Название второй команды.
        return: Словарь с количеством матчей, процентом побед team1 и средними голами team1.
        """

        if team2 is None:
            return {}

        # Фильтруем все матчи между двумя командами
        h2h = self.df[
            ((self.df['HomeTeam'] == team1) & (self.df['AwayTeam'] == team2)) |
            ((self.df['HomeTeam'] == team2) & (self.df['AwayTeam'] == team1))
        ]
        
        if h2h.empty:
            return {}
        
        team1_wins = len(h2h[((h2h['HomeTeam'] == team1) & (h2h['FTR'] == 'H')) | 
                        ((h2h['AwayTeam'] == team1) & (h2h['FTR'] == 'A'))])
        
        # Средние голы team1 в очных встречах
        avg_goals = h2h.apply(
            lambda x: x['FTHG'] if x['HomeTeam'] == team1 else x['FTAG'], axis=1).mean()
        avg_goals = 0 if pd.isnull(avg_goals) else avg_goals

        return {
            'HeadToHeadMatches': len(h2h),
            'HeadToHeadWinRate': (team1_wins / len(h2h)) * 100 if len(h2h) > 0 else 0,
            'HeadToHeadAvgGoals': avg_goals
        }