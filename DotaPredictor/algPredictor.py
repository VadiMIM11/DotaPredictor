import sys
import joblib
import numpy as np
import os
import json

from tqdm import tqdm

import DotaPredictor
import config
import preprocessing

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    if np.any((p <= 0) | (p >= 1)):  # Ensure p is in (0, 1) interval
        raise ValueError("Input probabilities must be between 0 and 1")
    return np.log(p / (1 - p))

def getWinrateWith(hero1, hero2, all_stats):
    # Ensure inputs are ints for dictionary lookup
    h1 = int(hero1)
    h2 = int(hero2)
    if h1 == 0 or h2 == 0: # Empty hero slot
        return None
    
    try:
        # Access: all_stats["with"][hero1][hero2]
        float_wr = all_stats["with"][h1][h2]
        return logit(float_wr)
    except KeyError:
        # If the hero pair doesn't exist in stats, return 0.5 (neutral)
        return logit(0.5)

def getWinrateAgainst(hero1, hero2, all_stats):
    h1 = int(hero1)
    h2 = int(hero2)
    if h1 == 0 or h2 == 0: # Empty hero slot
        return None

    try:
        # Access: all_stats["vs"][hero1][hero2]
        float_wr = all_stats["vs"][h1][h2]
        return logit(float_wr)
    except KeyError:
        return logit(0.5)
    
def get_avg_wr_with(radiantHeroes, direHeroes, all_stats):

    if len(radiantHeroes) != 5 or len(direHeroes) != 5:
        print("Invalid team format: ", radiantHeroes, direHeroes, file=sys.stderr)
        raise ValueError("Invalid team format. Must be length 5 (use 0 for empty slots)")

    wr_with_sum = 0.0
    countWith = 0
    for i in range(0, len(radiantHeroes)):
        if int(radiantHeroes[i]) == 0: 
            continue
        for j in range(i, len(radiantHeroes)):
            if int(radiantHeroes[j]) == 0:
                continue
            wr = getWinrateWith(radiantHeroes[i], radiantHeroes[j], all_stats)
            #print(f"Winrate of {radiantHeroes[i]} with {radiantHeroes[j]}: {wr}")
            if wr is not None:
                wr_with_sum += wr
                countWith += 1

    if countWith == 0:
        tqdm.write("Warning: No valid hero matchups found for with calculation. Empty draft?")
        return logit(0.5) # Neutral if no data

    avg_wr_with = wr_with_sum / countWith
    return avg_wr_with

def get_avg_wr_against(radiantHeroes, direHeroes, all_stats):

    if len(radiantHeroes) != 5 or len(direHeroes) != 5:
        print("Invalid team format: ", radiantHeroes, direHeroes, file=sys.stderr)
        raise ValueError("Invalid team format")

    wr_against_sum = 0.0
    countAgainst = 0
    for i in range(0, len(radiantHeroes)):
        if int(radiantHeroes[i]) == 0:
            continue
        for j in range(0, len(direHeroes)):
            if int(direHeroes[j]) == 0:
                continue
            wr = getWinrateAgainst(radiantHeroes[i], direHeroes[j], all_stats)
            #print(f"Winrate of {radiantHeroes[i]} vs {direHeroes[j]}: {wr}")
            if wr is not None:
                wr_against_sum += wr
                countAgainst += 1

    if countAgainst == 0:
        tqdm.write("Warning: No valid hero matchups found for against calculation. Empty draft?")
        return logit(0.5) # Neutral if no data        

    avg_wr_against = wr_against_sum / countAgainst
    return avg_wr_against

def predict(radiant_ids, dire_ids, all_stats):
    avg_wr_with = get_avg_wr_with(radiant_ids, dire_ids, all_stats)
    avg_wr_against = get_avg_wr_against(radiant_ids, dire_ids, all_stats)

    final_prediction = sigmoid((avg_wr_with + avg_wr_against) / 2.0)
    return final_prediction

def predict_by_match_id(m_id):
    try:
        with open(os.path.join("data", "all_hero_stats.json"), "r", encoding=config.DEFAULT_ENCODING) as f:
            all_stats = json.load(f)
    except IOError as e:
        print(f"Could not access hero stats file: {e}", file=sys.stderr)
        print("Try fetching hero stats from stratz using --update", file=sys.stderr)
        exit(1)
    all_stats = preprocessing.load_stats()
    match = DotaPredictor.get_match_by_id(m_id)
    feature_vector = preprocessing.generate_multihot_fv(match)
    return predict(feature_vector, all_stats)

