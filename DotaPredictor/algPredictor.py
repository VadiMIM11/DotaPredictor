import numpy as np
import os
import json

import DotaPredictor
import config

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    if np.any((p <= 0) | (p >= 1)):  # Ensure p is in (0, 1) interval
        raise ValueError("Input probabilities must be between 0 and 1")
    return np.log(p / (1 - p))

def getWinrateWith(hero1, hero2, all_stats):
    hero1_matchups = all_stats["data"]["heroStats"].get(f"hero{hero1}MatchUp")
    if not hero1_matchups:
        print(f"No data for hero {hero1}")
        exit(1)
    if hero1 == hero2:
        return logit(hero1_matchups[0]["winRate"])
    for entry in hero1_matchups:
        if entry["heroId"] == hero1:
            for with_entry in entry["with"]:
                if with_entry["heroId2"] == hero2:
                    return logit(with_entry["winsAverage"])
        else:
            print(f"Hero id mismatch: hero{hero1}MatchUp entry has heroId: {hero1}")
            exit(1)

def getWinrateAgainst(hero1, hero2, all_stats):
    hero1_matchups = all_stats["data"]["heroStats"].get(f"hero{hero1}MatchUp")
    if not hero1_matchups:
        print(f"No data for hero {hero1}")
        exit(1)
    for entry in hero1_matchups:
        if entry["heroId"] == hero1:
            for with_entry in entry["vs"]:
                if with_entry["heroId2"] == hero2:
                    return logit(with_entry["winsAverage"])
        else:
            print(f"Hero id mismatch: hero{hero1}MatchUp entry has heroId: {hero1}")
            exit(1)

def predict(feature_vector, all_stats):
    radiantHeroes = []
    direHeroes = []
    for i in range(1, config.MAX_HERO_ID+1):
        if feature_vector[i] == 1:
            radiantHeroes.append(i)
        elif feature_vector[i] == -1:
            direHeroes.append(i)
    
    if len(radiantHeroes) != 5 or len(direHeroes) != 5:
        print("Invalid feature vector: ", feature_vector)
        raise ValueError("Invalid feature vector")
        #exit(1)

    wr_with_sum = 0.0
    wr_against_sum = 0.0
    countWith = 0
    countAgainst = 0
    for i in range(0, len(radiantHeroes)):
        for j in range(i, len(radiantHeroes)):
            wr = getWinrateWith(radiantHeroes[i], radiantHeroes[j], all_stats)
            #print(f"Winrate of {radiantHeroes[i]} with {radiantHeroes[j]}: {wr}")
            wr_with_sum += wr
            countWith += 1

    if countWith == 0:
        raise ValueError("No 'with' data found")

    avg_wr_with = wr_with_sum / countWith

    for i in range(0, len(radiantHeroes)):
        for j in range(0, len(direHeroes)):
            wr = getWinrateAgainst(radiantHeroes[i], direHeroes[j], all_stats)
            #print(f"Winrate of {radiantHeroes[i]} vs {direHeroes[j]}: {wr}")
            wr_against_sum += wr
            countAgainst += 1

    if countAgainst == 0:
        raise ValueError("No 'against' data found")
    avg_wr_against = wr_against_sum / countAgainst

    final_prediction = sigmoid((avg_wr_with + avg_wr_against) / 2.0)
    return final_prediction

def predict_by_match_id(m_id):
    try:
        with open(os.path.join("data", "all_hero_stats.json"), "r", encoding=config.DEFAULT_ENCODING) as f:
            all_stats = json.load(f)
    except IOError as e:
        print(f"Could not access hero stats file: {e}")
        print("Try fetching hero stats from stratz using --update")
        exit(1)
    match = DotaPredictor.get_match_by_id(m_id)
    feature_vector = DotaPredictor.generate_feature_vector(match)
    return predict(feature_vector, all_stats)

