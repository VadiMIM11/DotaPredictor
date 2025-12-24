import json
import os
import sys
from joblib import dump
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import tqdm

import DotaPredictor 
import algPredictor
import config

ALL_STATS = None
def load_stats():
    global ALL_STATS
    if ALL_STATS is None:
        with open(os.path.join("data", "all_hero_stats.json"), "r") as f:
            ALL_STATS = json.load(f)
    return ALL_STATS

def extract_hero_ids_from_json(match_json):
    radiant_heroes = []
    dire_heroes = []

    if match_json.get("pickBans") is None:
        return [], []

    for pick in match_json["pickBans"]:
        if pick["isPick"] is True:
            # Return STRINGS because Gensim needs strings
            hero_id = str(pick["heroId"])
            
            if pick["isRadiant"]:
                radiant_heroes.append(hero_id)
            else:
                dire_heroes.append(hero_id)
                
    return radiant_heroes, dire_heroes

def load_raw_matches():
    try:
        path = os.path.join(config.DATA_FOLDER, "clean_train.json")
        with open(path, encoding=config.DEFAULT_ENCODING) as f:
            data = json.load(f)
        # Return the list of match dictionaries directly
        return data.get("data", {}).values()
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr)
        return []

def load_data_from_file():
    try:
        path = os.path.join(config.DATA_FOLDER, "clean_train.json")
        with open(path, encoding=config.DEFAULT_ENCODING) as f:
            data = json.load(f)
        return generate_treining_set(data) # Returns X, y
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return None, None


# def extract_hero_ids(feature_vector):
#     radiant_heroes = [i for i, val in enumerate(feature_vector) if val == 1]
#     dire_heroes = [i for i, val in enumerate(feature_vector) if val == -1]
#     return radiant_heroes, dire_heroes


def generate_feature_vector(match_json):
    model = DotaPredictor.load_embedding_model()
    r_ids, d_ids = extract_hero_ids_from_json(match_json)
    stats = load_stats()
    
    dim = model.vector_size
    rad_vec = np.zeros(dim)
    dire_vec = np.zeros(dim)
    
    for h_id in r_ids:
        if h_id in model.wv:
            rad_vec += model.wv[h_id]
            
    for h_id in d_ids:
        if h_id in model.wv:
            dire_vec += model.wv[h_id]

    avg_wr_with = algPredictor.get_avg_wr_with(r_ids, d_ids, stats)
    avg_wr_against = algPredictor.get_avg_wr_against(r_ids, d_ids, stats)
    stat_prob = algPredictor.sigmoid((avg_wr_with + avg_wr_against) / 2.0)
            
    return np.concatenate([rad_vec, dire_vec, [stat_prob], [avg_wr_with], [avg_wr_against]])

def get_label(match_json):
    if match_json.get("didRadiantWin") is None:
        print("No match result in match:", match_json["id"], file=sys.stderr)
        raise ValueError("No match result")
    if match_json["didRadiantWin"]:
        return 1
    else:
        return -1


def generate_treining_set(matches_json):
    matches = matches_json.get("data")
    X_list = []
    y_list = []
    for key, match in tqdm.tqdm(matches.items()):
        feature_vector = generate_feature_vector(match)
        label = get_label(match)
        X_list.append(feature_vector)
        y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)

    print("Scaling features...", file=sys.stderr)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if not os.path.exists(config.MODELS_FOLDER):
        os.makedirs(config.MODELS_FOLDER)
        print(f"Created folder: {config.MODELS_FOLDER}", file=sys.stderr)
        
    path = os.path.join(config.MODELS_FOLDER, "scaler.joblib")
    dump(scaler, path)
    print(f"Model saved in '{path}'", file=sys.stderr)

    return X, y


def train_and_save_embeddings():
    print("Loading raw match data...", file=sys.stderr)
    
    # 1. Load RAW matches 
    matches = load_raw_matches()
    
    sentences = []
    
    # 2. Extract IDs directly from JSON
    for match in matches:
        r_ids, d_ids = extract_hero_ids_from_json(match)
        
        # Add as two separate sentences (Team logic)
        if len(r_ids) == 5:
            sentences.append(r_ids)
        else:
            print(f"Warning: Skipping Radiant team with {len(r_ids)} heroes.", file=sys.stderr)
        if len(d_ids) == 5:
            sentences.append(d_ids)
        else:
            print(f"Warning: Skipping Dire team with {len(d_ids)} heroes.", file=sys.stderr)

    print(f"Training on {len(sentences)} teams...", file=sys.stderr)
    
    # 3. Train
    model = Word2Vec(sentences, vector_size=16, window=5, min_count=1, workers=4)
    
    # 4. Save
    if not os.path.exists(config.MODELS_FOLDER):
        os.makedirs(config.MODELS_FOLDER)
    model.save(os.path.join(config.MODELS_FOLDER, "embeddings.model"))
    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    train_and_save_embeddings()