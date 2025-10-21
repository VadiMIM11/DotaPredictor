import argparse
from ast import parse
import re
import numpy as np
import stratzQueries as sq
import json
import os
from tqdm import tqdm

import config

def extract_hero_ids(feature_vector):
    radiant_heroes = [i for i, val in enumerate(feature_vector) if val == 1]
    dire_heroes = [i for i, val in enumerate(feature_vector) if val == -1]
    return radiant_heroes, dire_heroes

def generate_feature_vector(match_json):
    feture_vector = np.zeros(config.MAX_HERO_ID + 1, dtype=int)
    if not match_json["pickBans"]:
        print("No pickBans in match:", match_json["id"])
        raise Exception("No pickBans in match")
        exit(1)
    for pick in match_json["pickBans"]:
        if pick["isPick"] is True:
            hero_id = pick["heroId"]
            if hero_id < 1 or hero_id > config.MAX_HERO_ID:
                print("Hero id out of range:", hero_id)
                raise Exception("Hero id out of range")
                exit(1)
            feture_vector[hero_id] = 1 if pick["isRadiant"] else -1
    return feture_vector

def get_label(match_json):
    if not match_json["didRadiantWin"]:
        print("No match result in match:", match_json["id"])
        raise Exception("No match result")
        exit(1)
    if match_json["didRadiantWin"]:
        return 1
    else:
        return -1

def generate_treining_set(matches_json):
    matches = matches.get("data")
    # TODO FINISH function 
    for match in matches:
        feture_vector = generate_feature_vector(match)
        label = get_label(match)
    return X, y

def update_local_stats(api_token):
    print("Updating local hero statistics dataset via stratz.com requests...")
    stats = sq.fetch_all_stats(api_token)

    FOLDER = config.DATA_FOLDER
    FILE_NAME = "all_hero_stats.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}")

    try:
        with open(PATH, "w") as f:
            json.dump(stats, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}")

    num_fetched = len(stats.get("data", {}).get("heroStats", {}))
    print(f"Fetched stats for {num_fetched} heroes")
    if(num_fetched != config.NUM_HEROES):
        print(f"Warning: Expected {config.NUM_HEROES} heroes, but got {num_fetched}.")

    print(f"Local hero stats dataset updated and saved to {PATH}")


def save_raw_train(data):
    FOLDER = config.DATA_FOLDER
    FILE_NAME = "raw_train.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}")
    try:
        with open(PATH, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}")

    print(f"Raw training dataset saved to {PATH}")


def clean_train():
    try:
        with open(os.path.join(config.DATA_FOLDER, "raw_train.json")) as f:
            data = json.load(f)
    except IOError as e:
        print(f"Could not access raw training data file: {e}")
        exit(1)

    FOLDER = config.DATA_FOLDER
    FILE_NAME = "clean_train.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}")

    matches = data["data"]
    print("Raw train size:", len(matches))
    filtered_matches = {}
    for match_key, match_value in matches.items():
        if (
            match_value
            and (match_value.get("lobbyType") == "RANKED" or not config.RANKED_ONLY)
            and (
                match_value.get("bracket") >= config.MIN_RANK
                and (match_value.get("gameMode") in config.GAMEMODES)
            )
        ):
            if match_value.get("pickBans") and len(match_value.get("pickBans")) > 0:
                filtered_pick_bans = [
                    pick for pick in match_value["pickBans"] if pick["isPick"] is True
                ]
            match_value["pickBans"] = filtered_pick_bans

            if match_value["pickBans"]:
                filtered_matches[match_key] = match_value
    print("Clean train size:", len(filtered_matches))
    data["data"] = filtered_matches

    try:
        with open(PATH, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}")

    print(f"Clean training dataset saved to {PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Welcome to the set of statistical and ML models to predict DOTA 2 match outcome"
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update local hero stats dataset via stratz.com requests",
        required=False,
    )
    parser.add_argument(
        "--stratz", type=str, help="Set path to file with stratz.com API token", required=False
    )
    parser.add_argument(
        "--fetch_train",
        action="store_true",
        help="Fetch latest matches for training dataset",
        required=False,
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter matches from raw_train.json based on settings in config.py",
        required=False,
    )
    parser.add_argument(
        "--test_alg_predict",
        action="store_true",
        required=False,
        help="Test algebraic predictor on training dataset",
        )

    args = parser.parse_args()

    if args.stratz:
        print("Stratz API token file path:", args.stratz)
        token = ""
        try:
            with open(args.stratz, "r") as f:
                token = f.read()
        except IOError as e:
            print(f"Error reading file: {e}")
        print("Stratz API Token: ", token[:10] + "...")
        config.API_TOKEN = token

    if args.update:
        if not args.stratz:
            print(
                "Please provide stratz.com API token to update local hero stats dataset using:\n --stratz <PATH>"
            )
            return

        update_local_stats(config.API_TOKEN)

    if args.fetch_train:
        if not args.stratz:
            print(
                "Please provide stratz.com API token to fetch latest matches for training dataset using:\n --stratz <PATH>"
            )
            return
        json_data = sq.fetch_train(config.api_token)
        save_raw_train(json_data)

    if args.filter:
        clean_train()

    if args.test_alg_predict:
        from algPredictor import predict
        # # Example feature vector with 5 radiant and 5 dire heroes
        # example_vector = np.zeros(config.MAX_HERO_ID + 1, dtype=int)
        # # Radiant heroes Medusa(94), Vengeful Spirit(20), Meepo(82), Enigma(33), Elder Titan(103)
        # example_vector[94] = 1
        # example_vector[20] = 1
        # example_vector[82] = 1
        # example_vector[33] = 1
        # example_vector[103] = 1
        # # Dire heroes Bloodseeker(4), Beastmaster(38), Doom(69), Batrider(65), Kez(145)
        # example_vector[4] = -1
        # example_vector[38] = -1
        # example_vector[69] = -1
        # example_vector[65] = -1
        # example_vector[145] = -1
        # try:
        #     win_prob = predict(example_vector)
        #     print(f"Predicted Radiant win probability: {win_prob:.2f}")
        # except Exception as e:
        #     print(f"Error during prediction: {e}")

        try:
            with open(os.path.join(config.DATA_FOLDER, "clean_train.json")) as f:
                data = json.load(f)
        except IOError as e:
            print(f"Could not access clean training data file: {e}")
            exit(1)
        try:
            with open(os.path.join("data", "all_hero_stats.json"), "r") as f:
                all_stats = json.load(f)
        except IOError as e:
            print(f"Could not access hero stats file: {e}")
            exit(1)
        matches = data["data"]
        total_matches = len(matches)
        print("Total matches to test:", total_matches)
        correct_predictions = 0
        total_predictions = 0
        true_positive_count = 0
        false_positive_count = 0
        true_negative_count = 0
        false_negative_count = 0
        outdrafted = []
        for match_key, match_value in tqdm(matches.items()):
            try:
                feature_vector = generate_feature_vector(match_value)
                predicted_prob = predict(feature_vector, all_stats)
                didRadiantWin = match_value.get("didRadiantWin")
                if didRadiantWin is None:
                    print(f"Match {match_key} has no radiantWin field, skipping...")
                    continue
                if abs(0.5 - predicted_prob) > 0.04:
                    # print()
                    # print(f"Highly outdrafted match detected!")
                    # print(f"Match ID: {match_key}")
                    # print(f"Predicted Radiant win probability: {predicted_prob:.2f}")
                    # print(f"Did Radiant win: {didRadiantWin}")
                    # radiant_heroes, dire_heroes = extract_hero_ids(feature_vector)
                    # print(f"Radiant Draft: {radiant_heroes}")
                    # print(f"Dire Draft: {dire_heroes}")
                    # print()
                    outdrafted.append(match_value.get("id"))
                predictRadiantWin = predicted_prob > 0.5
                if predictRadiantWin == didRadiantWin:
                    correct_predictions += 1
                    if didRadiantWin:
                        true_positive_count += 1
                    else:
                        true_negative_count += 1
                else:
                    if didRadiantWin:
                        false_negative_count += 1
                    else:
                        false_positive_count += 1
                total_predictions += 1
            except Exception as e:
                print(f"Error processing match {match_key}: {e}")
                continue
        print()
        print(f"Total predictions made: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"True Positives: {true_positive_count}")
        print(f"False Positives: {false_positive_count}")
        print(f"True Negatives: {true_negative_count}")
        print(f"False Negatives: {false_negative_count}")
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0 # Precision TP / (TP + FP)
        sensitivity = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0 # Sensitivity (Recall) TP / (TP + FN)
        specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0 # Specificity TN / (TN + FP)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

        print("HIghly Outdrafted Matches:", outdrafted)
if __name__ == "__main__":
    main()