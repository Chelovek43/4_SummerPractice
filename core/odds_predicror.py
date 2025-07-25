import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler


class OddsMatchPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.scaler = MinMaxScaler()
        self.features = ['odds_home', 'odds_draw', 'odds_away']
        self.prepare_data()
        
    def prepare_data(self):
        """Подготовка данных: создание целевой переменной и нормализация"""
        self.data['result'] = self.data.apply(
            lambda x: 1 if float(x['home_score']) > float(x['away_score']) else (
                0 if float(x['home_score']) == float(x['away_score']) else -1
            ), axis=1
        )
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        
    def train_model(self, test_size=0.2, random_state=42):
        """Обучение модели с комбинацией RF и ручных правил"""
        X = self.data[self.features]
        y = self.data['result']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=150,  
            max_depth=16,      
            random_state=42,
            class_weight={-1: 1, 0: 2, 1: 1},  
            min_samples_split=3, 
        )
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def predict_match(self, home_odd, draw_odd, away_odd):
        """Предсказание исхода матча по коэффициентам"""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train_model() first.")
            
        try:
            # Преобразуем в числа, заменяя запятые
            home = float(str(home_odd).replace(',', '.'))
            draw = float(str(draw_odd).replace(',', '.'))
            away = float(str(away_odd).replace(',', '.'))
            
            # Нормализация и предсказание
            input_data = self.scaler.transform([[home, draw, away]])
            proba = self.model.predict_proba(input_data)[0]

            # Определяем рекомендуемый исход
            max_prob_idx = np.argmax(proba)
            outcomes = ['Победа гостей', 'Ничья', 'Победа хозяев']
            recommended = {
                'outcome': outcomes[max_prob_idx],
                'probability': proba[max_prob_idx]
            }
            
            return {
                'home_win': proba[2],
                'draw': proba[1], 
                'away_win': proba[0],
                'recommended': recommended
            }
        except Exception as e:
            raise ValueError(f"Ошибка предсказания: {str(e)}")
    
    def get_outcome_name(self, prediction):
        """Преобразует численный результат в текстовый"""
        outcomes = {
            1: "Home Win",
            0: "Draw", 
            -1: "Away Win"
        }
        return outcomes.get(prediction, "Unknown")