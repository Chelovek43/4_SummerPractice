import requests
import pandas as pd
import os


from datetime import datetime
import json

import time


# Конфигурация API
API_KEY = "lhcuwv1jvp9rgakz"  # Используйте Публичный ключ (или мой ключ, забирайте, ограничение - 300 запросов/мин)
#MATCH_ID = ""  # Пример ID матча
BASE_URL = "https://api.sstats.net"
"""
Посмотрите ID матчей и укажите их ниже, лучше до 300 за раз, например
START_ID = 1208501 
END_ID = 1208835
"""
START_ID = 1208781
END_ID = 1208783


def fetch_match_data(match_id):
    """Получение данных одного матча"""
    url = f"{BASE_URL}/games/{match_id}"
    headers = {"apikey": API_KEY}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при запросе матча {match_id}: {str(e)}")
        return None

def process_match_data(match_json):
    """Извлечение всех нужных данных из JSON"""
    if not match_json or 'data' not in match_json:
        return None
    
    try:
        game = match_json['data']['game']
        odds = match_json['data'].get('odds', [])
        stats = match_json['data'].get('statistics', {})
        lineups = match_json['data'].get('lineups', {})

        # Функция для поиска коэффициентов
        def find_odds(market_id):
            for market in odds:
                if market['marketId'] == market_id:
                    return {o['name']: o['value'] for o in market['odds']}
            return {}

        # Получаем все нужные коэффициенты
        match_winner = find_odds(1)
        total_goals = find_odds(5)
        home_total = find_odds(16)
        away_total = find_odds(17)

        # Формируем полную запись матча
        match_record = {
            # Основная информация
            'match_id': game['id'],
            'date': game['date'],
            'status': game['status'],
            
            # Команды
            'home_team': game['homeTeam']['name'],
            'away_team': game['awayTeam']['name'],
            
            # Результаты
            'home_score': game['homeResult'],
            'away_score': game['awayResult'],
            'home_ht_score': game['homeHTResult'],
            'away_ht_score': game['awayHTResult'],
            
            # Лига
            'league': game['season']['league']['name'],
            'round': game['roundName'],
            
            # Коэффициенты
            'odds_home': match_winner.get('Home'),
            'odds_draw': match_winner.get('Draw'),
            'odds_away': match_winner.get('Away'),
            'over_1.5': total_goals.get('Over 1.5'),
            'under_1.5': total_goals.get('Under 1.5'),
            'home_over_0.5': home_total.get('Over 0.5'),
            'home_under_0.5': home_total.get('Under 0.5'),
            'away_over_0.5': away_total.get('Over 0.5'),
            'away_under_0.5': away_total.get('Under 0.5'),
            
            # Статистика
            'shots_on_target_home': stats.get('shotsOnGoalHome'),
            'shots_on_target_away': stats.get('shotsOnGoalAway'),
            'possession_home': stats.get('ballPossessionHome'),
            'possession_away': stats.get('ballPossessionAway'),
            'corners_home': stats.get('cornerKicksHome'),
            'corners_away': stats.get('cornerKicksAway'),
            
            # Составы
            'home_formation': lineups.get('homeFormation'),
            'away_formation': lineups.get('awayFormation'),
            'home_coach': lineups.get('homeCoach', {}).get('name'),
            'away_coach': lineups.get('awayCoach', {}).get('name')
        }
        
        return match_record
    
    except KeyError as e:
        print(f"Ошибка при обработке матча: отсутствует ключ {str(e)}")
        return None

def save_matches_range(start_id, end_id):
    start_time = time.time()  # Запоминаем время начала

    """Сохранение диапазона матчей в один файл"""
    all_matches = []
    processed_count = 0
    
    print(f"Начинаем обработку матчей с ID {start_id} по {end_id}")
    
    for match_id in range(start_id, end_id + 1):
        print(f"Обрабатываем матч ID: {match_id}", end='\r')
        
        match_json = fetch_match_data(match_id)
        match_data = process_match_data(match_json)
        
        if match_data:
            all_matches.append(match_data)
            processed_count += 1
    
    if not all_matches:
        print("\nНе удалось получить данные ни по одному матчу")
        return
    
    # Создаем DataFrame
    df = pd.DataFrame(all_matches)
    
    # Создаем папку если не существует
    output_dir = r'C:\Users\ADMIN\Desktop\Python'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'matches_{start_id}_to_{end_id}.csv')
    
    # Сохраняем в CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nГотово! Обработано матчей: {processed_count}")
    print(f"Результаты сохранены в: {output_file}")
    print(f"Всего полей в каждой записи: {len(df.columns)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nВремя выполнения: {execution_time:.2f} секунд")

'''
if __name__ == "__main__":
    save_matches_range(START_ID, END_ID)
'''
 