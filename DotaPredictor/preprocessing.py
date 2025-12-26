import json
import os
import sys
from joblib import dump
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm

import DotaPredictor
import algPredictor
import config

ALL_STATS = None

def load_stats():
    global ALL_STATS
    if ALL_STATS is not None:
        return ALL_STATS

    path = os.path.join("data", "all_hero_stats.json")
    print(f"Loading and optimizing hero stats from {path}...", file=sys.stderr)
    
    try:
        with open(path, "r") as f:
            raw_stats = json.load(f)
    except Exception as e:
        print(f"Error loading stats file: {e}", file=sys.stderr)
        return None

    # Initialize the Optimized Hash Table
    # Structure:
    # {
    #    "with": { hero_id_1: { hero_id_2: winrate, ... }, ... },
    #    "vs":   { hero_id_1: { hero_id_2: winrate, ... }, ... }
    # }
    optimized_stats = {
        "with": {},
        "vs": {}
    }

    # Extract the huge dictionary from the JSON wrapper
    hero_data_map = raw_stats.get("data", {}).get("heroStats", {})

    # Iterate through all possible heroes
    # range(1, 200) covers all current and near-future heroes safely
    for i in range(1, config.MAX_HERO_ID + 50):
        hero_id = int(i)
        
        # Pre-initialize inner dictionaries so lookups don't crash on "key not found"
        # They will just be empty dicts if no data exists
        optimized_stats["with"][hero_id] = {}
        optimized_stats["vs"][hero_id] = {}

        # The key in the JSON file (e.g., "hero1MatchUp")
        json_key = f"hero{hero_id}MatchUp"
        
        matchup_data_list = hero_data_map.get(json_key)
        
        # If data is missing or empty, skip
        if not matchup_data_list:
            continue
            
        # Stratz returns a list, but for specific IDs it usually has 1 entry
        entry = matchup_data_list[0]
        
        # 1. Optimize "WITH"
        # Convert list of objects -> Dictionary {id: wr}
        raw_with = entry.get("with") or []
        for w in raw_with:
            target_id = int(w["heroId2"])
            win_rate = float(w["winsAverage"])
            optimized_stats["with"][hero_id][target_id] = win_rate

        # 2. Optimize "VS"
        # Convert list of objects -> Dictionary {id: wr}
        raw_vs = entry.get("vs") or []
        for v in raw_vs:
            target_id = int(v["heroId2"])
            win_rate = float(v["winsAverage"])
            optimized_stats["vs"][hero_id][target_id] = win_rate

    #print(f"Hero stats optimization complete. Stats size: {sys.getsizeof(optimized_stats)} Bytes", file=sys.stderr)
    ALL_STATS = optimized_stats
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


def load_raw_matches_train():
    try:
        path = os.path.join(config.DATA_FOLDER, "clean_train.json")
        with open(path, encoding=config.DEFAULT_ENCODING) as f:
            data = json.load(f)
        
        matches = [match for match in data["data"].values() if match is not None] # convert to list

        raw_train, raw_test = train_test_split(
            matches, test_size=0.25, random_state=config.RANDOM_STATE
        )

        return raw_train
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr)
        raise


def load_data_from_file():
    try:
        path = os.path.join(config.DATA_FOLDER, "clean_train.json")
        with open(path, encoding=config.DEFAULT_ENCODING) as f:
            data = json.load(f)
        return generate_embedded_training_set(data)  # Returns X, y
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return None, None


# def extract_hero_ids(feature_vector):
#     radiant_heroes = [i for i, val in enumerate(feature_vector) if val == 1]
#     dire_heroes = [i for i, val in enumerate(feature_vector) if val == -1]
#     return radiant_heroes, dire_heroes


def generate_embedding(match_json):
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

    return np.concatenate(
        [rad_vec, dire_vec, [stat_prob], [avg_wr_with], [avg_wr_against]]
    )

def generate_multihot_fv(match_json):
    feture_vector = np.zeros(config.MAX_HERO_ID + 1, dtype=int)
    if match_json.get("pickBans") is None:
        print("No pickBans in match:", match_json["id"], file=sys.stderr)
        raise ValueError("No pickBans in match")
    for pick in match_json["pickBans"]:
        if pick["isPick"] is True:
            hero_id = pick["heroId"]
            if hero_id < 1 or hero_id > config.MAX_HERO_ID:
                print("Hero id out of range:", hero_id, file=sys.stderr)
                raise ValueError("Hero id out of range")
            feture_vector[hero_id] = 1 if pick["isRadiant"] else -1
    return feture_vector

def generate_multihot_training_set(matches_json):
    matches = matches_json.get("data")
    X_list = []
    y_list = []
    for key, match in tqdm.tqdm(matches.items()):
        feature_vector = generate_multihot_fv(match)
        label = get_label(match)
        X_list.append(feature_vector)
        y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)

    return X, y

def get_label(match_json):
    if match_json.get("didRadiantWin") is None:
        print("No match result in match:", match_json["id"], file=sys.stderr)
        raise ValueError("No match result")
    if match_json["didRadiantWin"]:
        return 1
    else:
        return -1

def save_training_data(X, y):
    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER)
    
    # Save X and y
    dump(X, os.path.join(config.DATA_FOLDER, 'X_data.joblib'))
    dump(y, os.path.join(config.DATA_FOLDER, 'y_data.joblib'))
    print("Training data saved to disk.", file=sys.stderr)

def load_training_data():
    try:
        X = joblib.load(os.path.join(config.DATA_FOLDER, 'X_data.joblib'))
        y = joblib.load(os.path.join(config.DATA_FOLDER, 'y_data.joblib'))
        print("Loaded training data (X, y) from disk.", file=sys.stderr)
        return X, y
    except (FileNotFoundError, OSError):
        print("Cached training data not found. Please regenerate.", file=sys.stderr)
        return None, None

def generate_embedded_training_set(matches_json):
    matches = matches_json.get("data")
    X_list = []
    y_list = []
    for key, match in tqdm.tqdm(matches.items()):
        feature_vector = generate_embedding(match)
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

    save_training_data(X, y)

    return X, y

def generate_pytorch_vector(match_json):
    r_ids, d_ids = extract_hero_ids_from_json(match_json)
    
    # Pad/Trim to 5
    r_vec = [int(x) for x in r_ids][:5] + [0]*(5-len(r_ids))
    d_vec = [int(x) for x in d_ids][:5] + [0]*(5-len(d_ids))
    
    avg_wr_with = algPredictor.get_avg_wr_with(r_ids, d_ids, load_stats())
    avg_wr_against = algPredictor.get_avg_wr_against(r_ids, d_ids, load_stats())
    stats_prediction = algPredictor.sigmoid((avg_wr_with + avg_wr_against) / 2.0)
    stats_components = [avg_wr_with, avg_wr_against, stats_prediction]
    
    # Result is now length: 5 + 5 + 3
    return np.array(r_vec + d_vec + stats_components)

def generate_pytorch_training_set(data):
    matches = data.get("data")
    X_list = []
    y_list = []
    
    print("Generating PyTorch training set...", file=sys.stderr)
    
    iterator = tqdm.tqdm(matches.items()) if matches else []
    
    for key, match in iterator:
        try:
            feature_vector = generate_pytorch_vector(match)
            label = get_label(match)
            
            X_list.append(feature_vector)
            y_list.append(label)
        except Exception:
            # Skip matches where algPredictor fails (e.g., missing hero stats)
            continue
            
    X_nn = np.array(X_list)
    y_nn = np.array(y_list)
    
    save_pytorch_data(X_nn, y_nn)
    
    return X_nn, y_nn

def save_pytorch_data(X, y):
    if not os.path.exists(config.DATA_FOLDER):
        os.makedirs(config.DATA_FOLDER)
    
    # Save with a distinct filename so it doesn't overwrite standard models
    dump(X, os.path.join(config.DATA_FOLDER, 'X_nn_data.joblib'))
    dump(y, os.path.join(config.DATA_FOLDER, 'y_nn_data.joblib'))
    print("PyTorch training data (X_nn, y_nn) saved to disk.", file=sys.stderr)

def load_pytorch_data():
    try:
        X = joblib.load(os.path.join(config.DATA_FOLDER, 'X_nn_data.joblib'))
        y = joblib.load(os.path.join(config.DATA_FOLDER, 'y_nn_data.joblib'))
        print("Loaded PyTorch training data (X_nn, y_nn) from disk.", file=sys.stderr)
        return X, y
    except (FileNotFoundError, OSError):
        # Fail silently so we can generate from JSON if needed
        return None, None



def train_and_save_embeddings():
    print("Loading raw match data...", file=sys.stderr)

    # 1. Load RAW matches
    matches = load_raw_matches_train()

    sentences = []

    # 2. Extract IDs directly from JSON
    for match in matches:
        r_ids, d_ids = extract_hero_ids_from_json(match)
        label = get_label(match)
        # If Radiant Won, add "WIN" token to Radiant team, "LOSS" to Dire team
        r_label = "WIN" if label == 1 else "LOSS"
        d_label = "LOSS" if label == 1 else "WIN"
        # Add as two separate sentences (Team logic)
        if len(r_ids) == 5:
            # [Hero1, Hero2, ..., Hero5, "WIN"]
            sentences.append(r_ids + [r_label]) 
        else:
            print(
                f"Warning: Skipping Radiant team with {len(r_ids)} heroes.",
                file=sys.stderr,
            )
        if len(d_ids) == 5:
            sentences.append(d_ids + [d_label])
        else:
            print(
                f"Warning: Skipping Dire team with {len(d_ids)} heroes.",
                file=sys.stderr,
            )

    print(f"Training on {len(sentences)} teams...", file=sys.stderr)

    # 3. Train
    model = Word2Vec(sentences, vector_size=16, window=6, min_count=1, workers=4)

    # 4. Save
    if not os.path.exists(config.MODELS_FOLDER):
        os.makedirs(config.MODELS_FOLDER)
    model.save(os.path.join(config.MODELS_FOLDER, "embeddings.model"))
    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    train_and_save_embeddings()