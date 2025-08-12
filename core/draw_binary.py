import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
import numpy as np

__all__ = [
    "add_features",
    "calculate_head_to_head",
    "DrawBinaryClassifier",
]

def add_features(df):
    """
    Добавляет необходимые признаки для классификации ничьих.
    """
    df['HomeWin'] = (df['FTR'] == 'H').astype(int)
    df['AwayWin'] = (df['FTR'] == 'A').astype(int)
    df['Draw'] = (df['FTR'] == 'D').astype(int)
    df["FTR_numeric"] = df["FTR"].map({"H": 1, "D": 3.4, "A": 0})

    df["HomeForm"] = (
        df.groupby("HomeTeam")["FTR_numeric"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["AwayForm"] = (
        df.groupby("AwayTeam")["FTR_numeric"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["HomeAttack"] = df.groupby("HomeTeam")["FTHG"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["AwayDefense"] = df.groupby("AwayTeam")["FTHG"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())

    df['HomeLast3Goals'] = df.groupby('HomeTeam')['FTHG'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayLast3Conceded'] = df.groupby('AwayTeam')['FTHG'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    df = calculate_head_to_head(df)

    for col in ['HomeAttack', 'AwayDefense', 'HomeLast3Goals', 'AwayLast3Conceded']:
        df[col] = df[col].fillna(df[col].mean())

    return df

def calculate_head_to_head(df, home_team=None, away_team=None):
    """
    Универсальный расчет статистики личных встреч.
    """
    if home_team is not None and away_team is not None:
        h2h_matches = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
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
        df['HeadToHeadWinRate'] = 0.5
        for i, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            prev_matches = df.iloc[:i]
            win_rate = calculate_head_to_head(prev_matches, home, away)
            df.at[i, 'HeadToHeadWinRate'] = win_rate
        return df

class DrawBinaryClassifier:
    """
    Класс для бинарной классификации ничьих в футбольных матчах.
    """
    def __init__(
        self,
        features=None,
        model_type="rf",
        oversample=False,
        random_state=42,
        class_weight={0: 1.0, 1: 1.75} 
    ):
        self.features = features
        self.model_type = model_type
        self.oversample = oversample
        self.random_state = random_state
        self.class_weight = class_weight
        self.model = None
        self.best_threshold = 0.5  # По умолчанию

    def fit(self, df, X_val=None, y_val=None, autotune_threshold=False, plot=False):
        """
        Обучение бинарного классификатора ничьих.
        Если autotune_threshold=True и переданы X_val, y_val, автоматически подбирает лучший threshold.
        """
        if self.features is None:
            self.features = [
                "HomeForm",
                "AwayForm",
                "HomeAttack",
                "AwayDefense",
                "HeadToHeadWinRate",
                "HomeLast3Goals",
                "AwayLast3Conceded",
            ]
        X = df[self.features]
        y = (df["FTR"] == "D").astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        if self.oversample:
            sm = SMOTE(random_state=self.random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        if self.model_type == "logreg":
            self.model = LogisticRegression(
                class_weight=self.class_weight, max_iter=1000, random_state=self.random_state
            )
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        else:
            raise ValueError("model_type должен быть 'logreg' или 'rf'")
        self.model.fit(X_train, y_train)
        # Метрики на тесте
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        print("=== Draw Binary Classifier ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

        # Автоматический подбор threshold
        if autotune_threshold and X_val is not None and y_val is not None:
            self.best_threshold, _ = self.tune_draw_threshold(self, X_val, y_val, plot=plot)
            #self.best_threshold = 0.43
            print(f"Автоматически выбранный threshold: {self.best_threshold:.2f}")

        """
        if plot:
                y_proba = self.predict_proba(X_val)
                plt.hist(y_proba[y_val == 1], bins=20, alpha=0.5, label='Draws')
                plt.hist(y_proba[y_val == 0], bins=20, alpha=0.5, label='Not Draws')
                plt.xlabel('Predicted draw probability')
                plt.ylabel('Count')
                plt.legend()
                plt.show()
        """

    def predict_proba(self, X):
        """
        Вероятность ничьей для новых данных.
        """
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X, threshold=None):
        """
        Предсказание ничьи (1) или не-ничьи (0) по порогу.
        Если threshold не указан, используется self.best_threshold.
        """
        if threshold is None:
            threshold = self.best_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @staticmethod
    def tune_draw_threshold(draw_clf, X_val, y_val, plot=True):
        """
        Перебирает значения порога для бинарного классификатора ничьих,
        считает метрики и строит confusion matrix для лучшего порога.
        """
        thresholds = np.linspace(0.05, 0.95, 37)
        best_f1 = 0
        best_threshold = 0.5
        metrics_dict = {}

        y_proba = draw_clf.predict_proba(X_val)

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            metrics_dict[thresh] = (acc, prec, rec, f1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        print(f"Лучший порог по F1-score: {best_threshold:.2f}, F1-score: {best_f1:.3f}")
        return best_threshold, metrics_dict
