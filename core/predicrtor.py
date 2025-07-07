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
    Класс для прогнозирования результатов футбольных матчей с использованием моделей Пуассона и Random Forest.
    
    Attributes:
        df (pd.DataFrame): DataFrame с историческими данными о матчах
        home_model (PoissonRegressor): Модель для предсказания голов домашней команды
        away_model (PoissonRegressor): Модель для предсказания голов гостевой команды
        features (list): Список используемых признаков для прогнозирования
        base_model (RandomForestClassifier): Основная модель Random Forest
        close_model (RandomForestClassifier): Модель для близких матчей
        rf_features (list): Список признаков для модели Random Forest
    """
    
    def __init__(self, data_path="football.csv"):
        """
        Инициализация прогнозиста.
        
        Args:
            data_path (str): Путь к файлу с данными о матчах
        """
        self.df = self.load_and_prepare_data(data_path)
        self.home_model, self.away_model, self.features = self.train_poisson_models()
        
        # Инициализация Random Forest моделей
        self.rf_features = [
            "HomeForm", "AwayForm", "HomeAttack", "AwayDefense",
            "HeadToHeadWinRate", "HomeLast3Goals", "AwayLast3Conceded"
        ]
        self.base_model, self.close_model = self.train_random_forest_models()
    
    def load_and_prepare_data(self, data_path):
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
        df = self.calculate_head_to_head(df)
        
        # Заполнение пропусков
        for col in ['HomeAttack', 'AwayDefense', 'HomeLast3Goals', 'AwayLast3Conceded']:
            df[col] = df[col].fillna(df[col].mean())
            
        return df
    
    def calculate_head_to_head(self, df):
        """
        Расчет истории личных встреч между командами.
        
        Args:
            df (pd.DataFrame): Исходный DataFrame
            
        Returns:
            pd.DataFrame: DataFrame с добавленным столбцом HeadToHeadWinRate
        """
        df['HeadToHeadWinRate'] = 0.5  # Значение по умолчанию
        
        for i, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Все предыдущие матчи между этими командами в два отдельных шага
            prev_matches = df.iloc[:i]  # Сначала берем срез по индексу
            prev_matches = prev_matches[
                (prev_matches['HomeTeam'] == home_team) & (prev_matches['AwayTeam'] == away_team) |
                (prev_matches['HomeTeam'] == away_team) & (prev_matches['AwayTeam'] == home_team)
            ]
            
            if len(prev_matches) > 0:
                win_rates = []
                for _, match in prev_matches.iterrows():
                    if match['HomeTeam'] == home_team:
                        # Домашняя команда дома: H=победа, D=ничья, A=поражение
                        win_rates.append(1 if match['FTR'] == 'H' else (0.5 if match['FTR'] == 'D' else 0))
                    else:
                        # Домашняя команда в гостях: A=победа, D=ничья, H=поражение
                        win_rates.append(1 if match['FTR'] == 'A' else (0.5 if match['FTR'] == 'D' else 0))
                
                df.at[i, 'HeadToHeadWinRate'] = np.mean(win_rates)
        
        return df
    
    def train_random_forest_models(self):
        """
        Обучение моделей Random Forest с выводом метрик.
        
        Returns:
            tuple: (base_model, close_model) - основная модель и модель для близких матчей
        """
        target = "FTR"
        
        X = self.df[self.rf_features]
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Основная модель
        print("\n" + "="*40)
        print("=== Обучение основной модели ===")
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight={'H': 1, 'D': 3, 'A': 1},
            random_state=42
        )
        base_model.fit(X_train, y_train)
        
        # Вывод метрик для основной модели
        y_pred_base = base_model.predict(X_test)
        print("\n=== Основная модель ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.6f}")
        print(classification_report(y_test, y_pred_base, digits=2))
        
        # Модель для близких матчей
        print("\n" + "="*40)
        print("=== Обучение модели для ничьих ===")
        close_matches_mask = (
            (abs(self.df['HomeForm'] - self.df['AwayForm']) < 0.27) &
            (abs(self.df['HomeAttack'] - self.df['AwayDefense']) < 0.37)
        )
        X_close = self.df.loc[close_matches_mask, self.rf_features]
        y_close = self.df.loc[close_matches_mask, target]
        
        X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(
            X_close, y_close, test_size=0.2, random_state=42
        )
        
        close_model = RandomForestClassifier(
            n_estimators=100,
            class_weight={'H': 1, 'D': 6, 'A': 1},
            random_state=42
        )
        close_model.fit(X_train_close, y_train_close)
        
        # Вывод метрик для модели ничьих
        y_pred_close = close_model.predict(X_test_close)
        print("\n=== Для ничьих модель ===")
        print(f"Accuracy: {accuracy_score(y_test_close, y_pred_close):.6f}")
        print(classification_report(y_test_close, y_pred_close, digits=2))
        
        # Комбинированная модель
        print("\n" + "="*40)
        print("=== Тест комбинированной модели ===")
        test_close_mask = (
            (abs(X_test['HomeForm'] - X_test['AwayForm']) < 0.27) &
            (abs(X_test['HomeAttack'] - X_test['AwayDefense']) < 0.37)
        )
        
        final_pred = [
            close_model.predict(X_test.iloc[[i]])[0] if test_close_mask.iloc[i]
            else y_pred_base[i]
            for i in range(len(X_test))
        ]
        
        print("\n=== Комбинированная модель ===")
        print(f"Accuracy: {accuracy_score(y_test, final_pred):.6f}")
        print(classification_report(y_test, final_pred, digits=2))
        
        return base_model, close_model
    
    def predict_with_rf(self, home_team, away_team):

        """Прогноз с использованием комбинированной модели Random Forest"""
        try:
            # Получаем последние данные команд с проверкой на существование
            home_matches = self.df[self.df['HomeTeam'] == home_team]
            away_matches = self.df[self.df['AwayTeam'] == away_team]
            
            if len(home_matches) == 0:
                raise ValueError(f"Нет данных о домашних матчах для команды {home_team}")
            if len(away_matches) == 0:
                raise ValueError(f"Нет данных о гостевых матчах для команды {away_team}")
                
            home_data = self.df[self.df['HomeTeam'] == home_team].iloc[-1]
            away_data = self.df[self.df['AwayTeam'] == away_team].iloc[-1]

            # Подготовка фичей для текущего матча
            match_features = pd.DataFrame([{
                'HomeForm': home_data['HomeForm'],
                'AwayForm': away_data['AwayForm'],
                'HomeAttack': home_data['HomeAttack'],
                'AwayDefense': away_data['AwayDefense'],
                'HeadToHeadWinRate': self.calculate_h2h_win_rate(home_team, away_team),
                'HomeLast3Goals': home_data['HomeLast3Goals'],
                'AwayLast3Conceded': away_data['AwayLast3Conceded']
            }])

            # Проверка, является ли матч "близким"
            is_close_match = (
                (abs(home_data['HomeForm'] - away_data['AwayForm']) < 0.27) &
                (abs(home_data['HomeAttack'] - away_data['AwayDefense']) < 0.37))
            
            # Выбор модели
            model = self.close_model if is_close_match else self.base_model
            model_name = "Модель для ничьих" if is_close_match else "Основная модель"

            # Прогноз
            probabilities = model.predict_proba(match_features[self.rf_features])[0]
            outcome = model.predict(match_features[self.rf_features])[0]

            return {
                'predicted_outcome': outcome,
                'probabilities': {
                    'home_win': probabilities[2],
                    'draw': probabilities[1],
                    'away_win': probabilities[0]
                },
                'model_used': model_name,
                'is_close_match': is_close_match
            }
            
        except Exception as e:
            print(f"Ошибка в predict_with_rf: {str(e)}")
            raise
    
    def calculate_h2h_win_rate(self, home_team, away_team):
        """
        Расчет коэффициента побед домашней команды в личных встречах.
        
        Args:
            home_team (str): Название домашней команды
            away_team (str): Название гостевой команды
            
        Returns:
            float: Коэффициент побед (0-1)
        """
        # Все матчи между командами (в любом порядке)
        h2h_matches = self.df[
            ((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
            ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))
        ]

        if len(h2h_matches) == 0:
            return 0.5  # Если нет истории

        win_rates = []
        for _, row in h2h_matches.iterrows():
            if row['HomeTeam'] == home_team:
                # Домашняя команда дома: H=победа, D=ничья, A=поражение
                win_rates.append(1 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0))
            else:
                # Домашняя команда в гостях: A=победа, D=ничья, H=поражение
                win_rates.append(1 if row['FTR'] == 'A' else (0.5 if row['FTR'] == 'D' else 0))

        return np.mean(win_rates)
    
    def train_poisson_models(self):
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
        likely_scores = self.simulate_scores(home_goals, away_goals, n_simulations)
        
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
    
    def simulate_scores(self, home_exp, away_exp, n=10000):
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
    
    


