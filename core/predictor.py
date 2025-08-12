import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from core.draw_binary import DrawBinaryClassifier, add_features

from sklearn.preprocessing import LabelEncoder # Кодирование категориальных признаков
from sklearn.preprocessing import StandardScaler # Нормализация числовых признаков
from sklearn.model_selection import train_test_split # Разделение данных на тренировочные и тестовые

# Для самой модели
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


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
        features = self.rf_features
        self.base_model, self.close_model = self.train_random_forest_models()
    
    def load_and_prepare_data(self, data_path):
        """
        Загрузка и подготовка данных без data leakage.
        """
        df = pd.read_csv(data_path)
        df = df.sort_values("Date").reset_index(drop=True)

        # Создаем необходимые столбцы для результатов
        df['HomeWin'] = (df['FTR'] == 'H').astype(int)
        df['AwayWin'] = (df['FTR'] == 'A').astype(int)
        df['Draw'] = (df['FTR'] == 'D').astype(int)
        df["FTR_numeric"] = df["FTR"].map({"H": 1, "D": 0.5, "A": 0})

        # Rolling-признаки только по прошлым матчам (shift(1)), теперь через .transform
        df["HomeForm"] = (
            df.groupby("HomeTeam")["FTR_numeric"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df["AwayForm"] = (
            df.groupby("AwayTeam")["FTR_numeric"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df["HomeAttack"] = (
            df.groupby("HomeTeam")["FTHG"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df["AwayDefense"] = (
            df.groupby("AwayTeam")["FTHG"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df["HomeLast3Goals"] = (
            df.groupby("HomeTeam")["FTHG"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df["AwayLast3Conceded"] = (
            df.groupby("AwayTeam")["FTHG"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

        # История личных встреч только по прошлым матчам
        df = self.calculate_head_to_head(df)

        # Заполнение пропусков
        for col in ['HomeAttack', 'AwayDefense', 'HomeLast3Goals', 'AwayLast3Conceded', 'HomeForm', 'AwayForm']:
            df[col] = df[col].fillna(df[col].mean())

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
        #print("Количество close matches:", close_matches_mask.sum(), "из", len(df))
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
    
    
    
    def calculate_head_to_head(self, df, home_team=None, away_team=None):
        """
        Универсальный расчет head-to-head только по прошлым матчам.
        """
        if home_team is not None and away_team is not None:
            # Для одного матча (например, будущего)
            h2h_matches = df[
                (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                 ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)))
                & (df['FTR'].notna())
            ]
            if len(h2h_matches) == 0:
                return 0.5
            win_rates = []
            for _, row in h2h_matches.iterrows():
                if row['HomeTeam'] == home_team:
                    win_rates.append(1 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0))
                else:
                    win_rates.append(1 if row['FTR'] == 'A' else (0.5 if row['FTR'] == 'D' else 0))
            return np.mean(win_rates)
        else:
            # Для всего DataFrame
            df = df.copy()
            df['HeadToHeadWinRate'] = 0.5
            for i, row in df.iterrows():
                home = row['HomeTeam']
                away = row['AwayTeam']
                prev_matches = df.iloc[:i]
                win_rate = self.calculate_head_to_head(prev_matches, home, away)
                df.at[i, 'HeadToHeadWinRate'] = win_rate
            return df
    
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
    
    def predict_with_rf(self, home_team, away_team, match_date=None):
        """
        Прогноз для будущего матча с использованием честных rolling-признаков и head-to-head.
        """
        try:
            match_features = self.get_last_features_for_match(home_team, away_team, match_date)
            is_close_match = (
                abs(match_features["HomeForm"].iloc[0] - match_features["AwayForm"].iloc[0]) < 0.27 and
                abs(match_features["HomeAttack"].iloc[0] - match_features["AwayDefense"].iloc[0]) < 0.37
            )
            model = self.close_model if is_close_match else self.base_model
            model_name = "Модель для ничьих" if is_close_match else "Основная модель"
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

    def predict_match(self, home_team, away_team, match_date=None, n_simulations=10000):
        """
        Прогноз для будущего матча с использованием Пуассона (rolling-признаки только по прошлым матчам).
        """
        match_features = self.get_last_features_for_match(home_team, away_team, match_date)
        home_goals = self.home_model.predict(match_features[self.features])[0]
        away_goals = self.away_model.predict(match_features[self.features])[0]
        home_win_prob, draw_prob, away_win_prob = self.calculate_outcome_probabilities(
            home_goals, away_goals)
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
    
    def calculate_outcome_probabilities(self, home_exp, away_exp, max_goals=10):
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
    
    def get_last_features_for_match(predictor, home_team, away_team, match_date=None):
        """
        Получить rolling-признаки и head-to-head для будущего матча.
        Если match_date не указана, берется конец истории.
        """
        df_hist = predictor.df
        if match_date is not None:
            df_hist = df_hist[df_hist["Date"] < match_date]

        # Последние значения rolling-признаков для обеих команд
        home_hist = df_hist[df_hist["HomeTeam"] == home_team]
        away_hist = df_hist[df_hist["AwayTeam"] == away_team]
        if home_hist.empty or away_hist.empty:
            raise ValueError("Недостаточно истории для одной из команд")

        home_data = home_hist.iloc[-1]
        away_data = away_hist.iloc[-1]

        # Head-to-head только по прошлым матчам
        h2h = predictor.calculate_head_to_head(df_hist, home_team, away_team)

        features = {
            "HomeForm": home_data["HomeForm"],
            "AwayForm": away_data["AwayForm"],
            "HomeAttack": home_data["HomeAttack"],
            "AwayDefense": away_data["AwayDefense"],
            "HeadToHeadWinRate": h2h,
            "HomeLast3Goals": home_data["HomeLast3Goals"],
            "AwayLast3Conceded": away_data["AwayLast3Conceded"],
        }
        return pd.DataFrame([features])
    
    def combined_predict(self, draw_clf, home_team, away_team, match_date, draw_threshold=None):
        """
        Комбинированный прогноз: если вероятность ничьей по бинарному классификатору выше порога,
        то прогнозируется ничья, иначе используется результат RF-модели.

        Args:
            draw_clf (DrawBinaryClassifier): обученный бинарный классификатор ничьих
            home_team (str): название домашней команды
            away_team (str): название гостевой команды
            match_date (str или None): дата матча
            draw_threshold (float или None): порог вероятности ничьей (если None — берётся оптимальный)

        Returns:
            dict: {
                'predicted_outcome': str,
                'draw_proba': float,
                'rf_probabilities': dict,
                'model_used': str
            }
        """
        X_upcoming = self.get_last_features_for_match(home_team, away_team, match_date)
        draw_proba = draw_clf.predict_proba(X_upcoming)[0]
        threshold = draw_clf.best_threshold if draw_threshold is None else draw_threshold

    
        print("home_team:", home_team)
        print("away_team:", away_team)
        print("match_date:", match_date)
        print("X_upcoming:", X_upcoming)
        print("draw_proba:", draw_proba)
        print("threshold:", threshold)

        if draw_proba > threshold:
            return {
                'predicted_outcome': 'D',
                'draw_proba': draw_proba,
                'rf_probabilities': None,
                'model_used': 'DrawBinaryClassifier'
            }
        else:
            rf_pred = self.predict_with_rf(home_team, away_team, match_date)
            return {
                'predicted_outcome': rf_pred['predicted_outcome'],
                'draw_proba': draw_proba,
                'rf_probabilities': rf_pred['probabilities'],
                'model_used': rf_pred['model_used']
            }
        
    @staticmethod
    def find_best_close_thresholds(
        df,
        rf_features,
        base_model,
        close_model,
        target_col="FTR",
        metric="accuracy",
        form_range=(0.05, 0.5, 10),
        attack_range=(0.05, 0.5, 10),
    ):
        """
        Перебирает значения порогов для определения close match и выбирает лучшие по заданной метрике.
        """
        best_score = -np.inf
        best_form_thr = None
        best_attack_thr = None

        form_thresholds = np.linspace(*form_range)
        attack_thresholds = np.linspace(*attack_range)

        X = df[rf_features]
        y = df[target_col]

        total = len(form_thresholds) * len(attack_thresholds)
        counter = 0

        for form_thr in form_thresholds:
            for attack_thr in attack_thresholds:
                counter += 1
                print(f"Перебор: {counter}/{total} (form_thr={form_thr:.3f}, attack_thr={attack_thr:.3f})")
                preds = []
                for i, row in X.iterrows():
                    is_close = (
                        abs(row["HomeForm"] - row["AwayForm"]) < form_thr and
                        abs(row["HomeAttack"] - row["AwayDefense"]) < attack_thr
                    )
                    model = close_model if is_close else base_model
                    pred = model.predict(pd.DataFrame([row]))[0]
                    preds.append(pred)
                if metric == "accuracy":
                    score = accuracy_score(y, preds)
                elif metric == "f1":
                    score = f1_score(y, preds, average="macro")
                else:
                    raise ValueError("Unknown metric")
                if score > best_score:
                    best_score = score
                    best_form_thr = form_thr
                    best_attack_thr = attack_thr

        print(f"Лучшие пороги: form={best_form_thr:.3f}, attack={best_attack_thr:.3f}, {metric}={best_score:.4f}")
        return best_form_thr, best_attack_thr, best_score
    
