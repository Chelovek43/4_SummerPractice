import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder # Кодирование категориальных признаков
from sklearn.preprocessing import StandardScaler # Нормализация числовых признаков
from sklearn.model_selection import train_test_split # Разделение данных на тренировочные и тестовые

# Для самой модели
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve, auc, average_precision_score


from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


class FootballMatchPredictor:
    """
    Класс для прогнозирования результатов футбольных матчей с использованием моделей Пуассона.
    
    Attributes:
        df (pd.DataFrame): DataFrame с историческими данными о матчах
        home_model (PoissonRegressor): Модель для предсказания голов домашней команды
        away_model (PoissonRegressor): Модель для предсказания голов гостевой команды
        features (list): Список используемых признаков для прогнозирования
    """
    
    def __init__(self, data_path="football.csv"):
        """
        Инициализация прогнозиста.
        
        Args:
            data_path (str): Путь к файлу с данными о матчах
        """
        self.df = self._load_and_prepare_data(data_path)
        self.home_model, self.away_model, self.features = self._train_poisson_models()
    
    def _load_and_prepare_data(self, data_path):
        """
        Загрузка и подготовка данных.
        
        Args:
            data_path (str): Путь к файлу с данными
            
        Returns:
            pd.DataFrame: Подготовленный DataFrame
        """
        df = pd.read_csv(data_path)
        
        # Создаем временный числовой столбец для результатов
        df["FTR_numeric"] = df["FTR"].map({"H": 1, "D": 0.5, "A": 0})

        # Форма команд (последние 5 матчей)
        df["HomeForm"] = (
            df.groupby("HomeTeam")["FTR_numeric"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True))
        
        df["AwayForm"] = (
            df.groupby("AwayTeam")["FTR_numeric"]
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True))
        
        # Средние показатели атаки и защиты
        df["HomeAttack"] = df.groupby("HomeTeam")["FTHG"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["AwayDefense"] = df.groupby("AwayTeam")["FTHG"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        
        # Последние 3 матча
        df['HomeLast3Goals'] = df.groupby('HomeTeam')['FTHG'].rolling(3).mean().reset_index(level=0, drop=True)
        df['AwayLast3Conceded'] = df.groupby('AwayTeam')['FTHG'].rolling(3).mean().reset_index(level=0, drop=True)
        
        # История личных встреч
        df = self._calculate_head_to_head(df)
        
        # Заполнение пропусков
        for col in ['HomeAttack', 'AwayDefense', 'HomeLast3Goals', 'AwayLast3Conceded']:
            df[col] = df[col].fillna(df[col].mean())
            
        return df
    
    def _calculate_head_to_head(self, df):
        """
        Расчет статистики личных встреч между командами.
        
        Args:
            df (pd.DataFrame): Исходный DataFrame
            
        Returns:
            pd.DataFrame: DataFrame с добавленной статистикой
        """
        df["HeadToHeadWinRate"] = 0.5  # Значение по умолчанию
        
        for index, row in df.iterrows():
            last_meetings = df[
                ((df["HomeTeam"] == row["HomeTeam"]) & (df["AwayTeam"] == row["AwayTeam"])) |
                ((df["HomeTeam"] == row["AwayTeam"]) & (df["AwayTeam"] == row["HomeTeam"]))
            ].head(5)
            
            if not last_meetings.empty:
                home_wins = last_meetings[last_meetings["FTR"] == "H"].shape[0]
                df.at[index, "HeadToHeadWinRate"] = home_wins / len(last_meetings)
                
        return df
    
    def _train_poisson_models(self):
        """
        Обучение моделей Пуассона для голов домашней и гостевой команд.
        
        Returns:
            tuple: (home_model, away_model, features)
        """
        features = ['HomeAttack', 'AwayDefense', 'HomeLast3Goals', 'AwayLast3Conceded']
        
        # Модель для голов хозяев
        home_model = PoissonRegressor()
        home_model.fit(self.df[features], self.df['FTHG'])
        
        # Модель для голов гостей
        away_model = PoissonRegressor()
        away_model.fit(self.df[features], self.df['FTAG'])
        
        return home_model, away_model, features
    
    def predict_match(self, home_team, away_team, n_simulations=10000):
        """
        Прогноз для конкретного матча.
        
        Args:
            home_team (str): Название домашней команды
            away_team (str): Название гостевой команды
            n_simulations (int): Количество симуляций для Монте-Карло
            
        Returns:
            dict: Результаты прогноза с ожидаемым счетом, вероятностями исходов и вероятными счетами
            
        Raises:
            ValueError: Если одна из команд не найдена в данных
        """
        # Проверка наличия команд в данных
        if home_team not in self.df['HomeTeam'].values:
            raise ValueError(f"Домашняя команда '{home_team}' не найдена в данных")
        if away_team not in self.df['AwayTeam'].values:
            raise ValueError(f"Гостевая команда '{away_team}' не найдена в данных")
        
        # Получаем последние данные команд
        home_data = self.df[self.df['HomeTeam'] == home_team].iloc[-1]
        away_data = self.df[self.df['AwayTeam'] == away_team].iloc[-1]
        
        # Подготовка признаков
        match_features = pd.DataFrame([{
            'HomeAttack': home_data['HomeAttack'],
            'AwayDefense': away_data['AwayDefense'],
            'HomeLast3Goals': home_data['HomeLast3Goals'],
            'AwayLast3Conceded': away_data['AwayLast3Conceded']
        }])
        
        # Прогноз ожидаемых голов
        home_goals = self.home_model.predict(match_features[self.features])[0]
        away_goals = self.away_model.predict(match_features[self.features])[0]
        
        # Расчет вероятностей исходов
        home_win_prob, draw_prob, away_win_prob = self._calculate_outcome_probabilities(
            home_goals, away_goals)
        
        # Моделирование вероятных счетов
        likely_scores = self._simulate_scores(home_goals, away_goals, n_simulations)
        
        return {
            'teams': {
                'home': home_team,
                'away': away_team
            },
            'expected_score': {
                'home': round(home_goals, 2),
                'away': round(away_goals, 2)
            },
            'outcome_probabilities': {
                'home_win': round(home_win_prob, 3),
                'draw': round(draw_prob, 3),
                'away_win': round(away_win_prob, 3)
            },
            'likely_scores': likely_scores
        }
    
    def _calculate_outcome_probabilities(self, home_exp, away_exp, max_goals=10):
        """
        Расчет вероятностей исходов матча.
        
        Args:
            home_exp (float): Ожидаемое количество голов домашней команды
            away_exp (float): Ожидаемое количество голов гостевой команды
            max_goals (int): Максимальное количество голов для расчета
            
        Returns:
            tuple: (вероятность победы хозяев, вероятность ничьи, вероятность победы гостей)
        """
        home_win_prob = np.sum([poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
                             for i in range(0, max_goals) for j in range(0, i)])
        
        draw_prob = np.sum([poisson.pmf(i, home_exp) * poisson.pmf(i, away_exp)
                         for i in range(0, max_goals)])
        
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return home_win_prob, draw_prob, away_win_prob
    
    def _simulate_scores(self, home_exp, away_exp, n=10000):
        """
        Моделирование вероятных счетов методом Монте-Карло.
        
        Args:
            home_exp (float): Ожидаемое количество голов домашней команды
            away_exp (float): Ожидаемое количество голов гостевой команды
            n (int): Количество симуляций
            
        Returns:
            dict: Словарь с вероятными счетами и их вероятностями
        """
        home_goals = poisson.rvs(home_exp, size=n)
        away_goals = poisson.rvs(away_exp, size=n)
        
        score_probs = pd.crosstab(home_goals, away_goals).div(n).stack()
        return score_probs.sort_values(ascending=False).head(10).to_dict()
    
    def get_team_list(self):
        """
        Возвращает список всех уникальных команд в данных.
        
        Returns:
            tuple: (home_teams, away_teams) - списки домашних и гостевых команд
        """
        home_teams = sorted(self.df['HomeTeam'].unique().tolist())
        away_teams = sorted(self.df['AwayTeam'].unique().tolist())
        return home_teams, away_teams


