import argparse
from ast import parse
import random
import re
from typing import Required
from urllib import request
import numpy as np
import algPredictor
import stratzQueries as sq
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
import logRegPredictor
import rbfPredictor
import mlpPredictor


def extract_hero_ids(feature_vector):
    radiant_heroes = [i for i, val in enumerate(feature_vector) if val == 1]
    dire_heroes = [i for i, val in enumerate(feature_vector) if val == -1]
    return radiant_heroes, dire_heroes


def generate_feature_vector(match_json):
    feture_vector = np.zeros(config.MAX_HERO_ID + 1, dtype=int)
    if match_json.get("pickBans") is None:
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
    if match_json.get("didRadiantWin") is None:
        print("No match result in match:", match_json["id"])
        raise Exception("No match result")
        exit(1)
    if match_json["didRadiantWin"]:
        return 1
    else:
        return -1


def generate_treining_set(matches_json):
    matches = matches_json.get("data")
    X_list = []
    y_list = []
    for key, match in matches.items():
        feature_vector = generate_feature_vector(match)
        label = get_label(match)
        X_list.append(feature_vector)
        y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)
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
    if num_fetched != config.NUM_HEROES:
        print(f"Warning: Expected {config.NUM_HEROES} heroes, but got {num_fetched}.")

    print(f"Local hero stats dataset updated and saved to {PATH}")


def get_match_by_id(match_id):
    try:
        with open(f"{config.DATA_FOLDER}/clean_train.json", "r") as f:
            matches = json.load(f)
        matches = matches.get("data")
        if matches is None:
            print("Error reading data file")
            return None
        for key, match in matches.items():
            current_id = match.get("id")
            if not current_id is None and current_id == match_id:
                return match
    except IOError:
        print("Data file not found. Please run the data fetching first.")
        # return None

    match = sq.fetch_match_by_id(config.API_TOKEN, match_id)
    if not match.get("data") is None:
        return match.get("data")["match"]
    else:
        return None


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


def predict_by_id(m_id):
    # m_id = 8525383837
    print(f"Precict match {m_id}")
    prob = rbfPredictor.predict_by_match_id(m_id)
    print(f"RBF: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%")
    prob = logRegPredictor.predict_by_match_id(m_id)
    print(f"LogReg: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%")
    prob = algPredictor.predict_by_match_id(m_id)
    print(f"Stats: {m_id} -> Radiant Win with {(prob * 100):.4f}%")
    print()


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
        "--stratz",
        type=str,
        help="Set path to file with stratz.com API token",
        required=False,
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

    parser.add_argument(
        "--train_logreg",
        action="store_true",
        required=False,
        help="Train logistic regression model on training dataset",
    )

    parser.add_argument(
        "--train_rbf",
        action="store_true",
        required=False,
        help="Train RBF SVM model on training dataset",
    )

    parser.add_argument(
        "--predict",
        type=int,
        required=False,
        help="Predict match outcome by match ID using all models",
    )

    parser.add_argument(
        "--test_prob_acc",
        action="store_true",
        required=False,
        help="Test probability accuracy of models on training dataset",
    )

    parser.add_argument(
        "--train_mlp",
        action="store_true",
        required=False,
        help="Train MLP model on training dataset",
    )

    parser.add_argument(
        "--generate_train",
        action="store_true",
        required=False,
        help="Generate training and testing sets from clean_train.json",)

    args = parser.parse_args()

    if args.stratz:
        print("Stratz API token file path:", args.stratz)
        token = ""
        try:
            with open(args.stratz, "r") as f:
                token = f.read()
        except IOError as e:
            print(f"Error reading file: {e}")
            exit(1)
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
        json_data = sq.fetch_train(config.API_TOKEN)
        save_raw_train(json_data)


    if args.filter:
        clean_train()

    def get_confidence_accuracy(probas, y_test, confidence_threshold):
        hit = 0
        miss = 0
        for i in range(len(probas)):
            predicted_prob = probas[i][1]
            predicted = 1 if predicted_prob >= 0.5 else -1
            actual = y_test[i]
            if abs(0.5 - predicted_prob) >= confidence_threshold:
                if actual == predicted:
                    hit += 1
                else:
                    miss += 1
        accuracy = hit / (hit + miss) if (hit + miss) > 0 else 0
        count = hit + miss
        return count, accuracy

    global X_train, X_test, y_train, y_test

    if args.train_rbf or args.train_logreg or args.train_mlp or args.generate_train:
        try:
            with open(os.path.join(config.DATA_FOLDER, "clean_train.json")) as f:
                data = json.load(f)
        except:
            print(
                "Could not access clean training data file. Please run with --filter first."
            )
            return

        global X_train, X_test, y_train, y_test
        X, y = generate_treining_set(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=config.RANDOM_STATE
        )

    if args.train_logreg:
        logRegPredictor.train_logreg(X_train, y_train)
        print()
        header = "="*5 + " Logistic Regression Predictor " + "="*5
        print(header)
        print("Total matches to test:", y_test.size)
        print()
        accuracy, cm = logRegPredictor.evaluate_logreg(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        # print("True Negatives:", cm[0][0])
        # print("False Positives:", cm[0][1])
        # print("False Negatives:", cm[1][0])
        # print("True Positives:", cm[1][1])
        precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0 # Precision TP / (TP + FP)
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0 # Sensitivity (Recall) TP / (TP + FN)
        specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0 # Specificity TN / (TN + FP)
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("=" * len(header))
        print()
        

    if args.train_rbf:
        print("Training RBF SVM Predictor...")
        rbfPredictor.train_rbf(X_train, y_train)
        print("Done training!")
        print()
        header = "="*5 + " RBF SVM Predictor " + "="*5
        print(header)
        print("Total matches to test:", y_test.size)
        print()
        accuracy, cm = rbfPredictor.evaluate_rbf(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        # print("True Negatives:", cm[0][0])
        # print("False Positives:", cm[0][1])
        # print("False Negatives:", cm[1][0])
        # print("True Positives:", cm[1][1])
        precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("=" * len(header))
        print()

    if args.train_mlp:
        print("Training MLP Predictor...")
        mlpPredictor.train_mlp(X_train, y_train)
        print("Done training!")
        print()
        header = "=" * 5 + " MLP Predictor " + "=" * 5
        print(header)
        print("Total matches to test:", y_test.size)
        print()
        accuracy, cm = mlpPredictor.evaluate_mlp(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        # print("True Negatives:", cm[0][0])
        # print("False Positives:", cm[0][1])
        # print("False Negatives:", cm[1][0])
        # print("True Positives:", cm[1][1])
        precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        sensitivity = (
            cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        )
        specificity = (
            cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        )
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("=" * len(header))
        print()

    if args.test_alg_predict:
        from algPredictor import predict

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
            print("Try fetching hero stats from stratz using --update")
            exit(1)

        print()
        header = "=" * 5 + " Statistical Prediictor " + "=" * 5
        print(header)
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
        for match_key, match_value in matches.items():
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
        # print(f"Total predictions made: {total_predictions}")
        # print(f"Correct predictions: {correct_predictions}")
        # print(f"True Positives: {true_positive_count}")
        # print(f"False Positives: {false_positive_count}")
        # print(f"True Negatives: {true_negative_count}")
        # print(f"False Negatives: {false_negative_count}")
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        precision = (
            true_positive_count / (true_positive_count + false_positive_count)
            if (true_positive_count + false_positive_count) > 0
            else 0
        )  # Precision TP / (TP + FP)
        sensitivity = (
            true_positive_count / (true_positive_count + false_negative_count)
            if (true_positive_count + false_negative_count) > 0
            else 0
        )  # Sensitivity (Recall) TP / (TP + FN)
        specificity = (
            true_negative_count / (true_negative_count + false_positive_count)
            if (true_negative_count + false_positive_count) > 0
            else 0
        )  # Specificity TN / (TN + FP)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

        # print("HIghly Outdrafted Matches:", outdrafted)
        print("=" * len(header))
        print()

    if args.test_prob_acc:
        header = "=" * 5 + " Probability Accuracy Test " + "=" * 5
        print(header)

        print("Statistical Predictor:")
        try:
            with open(os.path.join("data", "all_hero_stats.json"), "r") as f:
                all_stats = json.load(f)
        except IOError as e:
            print(f"Could not access hero stats file: {e}")
            print("Try fetching hero stats from stratz using --update")
            exit(1)

        stats_probs = []
        for i in range(len(X_test)):
            prob = algPredictor.predict(X_test[i], all_stats)
            stats_probs.append([1 - prob, prob])

        count50, stats50 = get_confidence_accuracy(stats_probs, y_test, 0.00)
        count55, stats55 = get_confidence_accuracy(stats_probs, y_test, 0.05)
        count60, stats60 = get_confidence_accuracy(stats_probs, y_test, 0.10)
        count65, stats65 = get_confidence_accuracy(stats_probs, y_test, 0.15)
        print(
            f"Stats accuracy with >=50% confidence: {stats50*100:.4f}% count: {count50}"
        )
        print(
            f"Stats accuracy with >=55% confidence: {stats55*100:.4f}% count: {count55}"
        )
        print(
            f"Stats accuracy with >=60% confidence: {stats60*100:.4f}% count: {count60}"
        )
        print(
            f"Stats accuracy with >=65% confidence: {stats65*100:.4f}% count: {count65}"
        )
        print()

        if args.train_logreg:
            print("Logistic Regression Predictor:")
            # logRegPredictor.train_logreg(X_train, y_train)
            logreg_probas = logRegPredictor.predict_proba_logreg(X_test)

            count50, logreg50 = get_confidence_accuracy(logreg_probas, y_test, 0.00)
            count55, logreg55 = get_confidence_accuracy(logreg_probas, y_test, 0.05)
            count60, logreg60 = get_confidence_accuracy(logreg_probas, y_test, 0.10)
            count65, logreg65 = get_confidence_accuracy(logreg_probas, y_test, 0.15)
            print(
                f"LogReg accuracy with >=50% confidence: {logreg50*100:.4f}% count: {count50}"
            )
            print(
                f"LogReg accuracy with >=55% confidence: {logreg55*100:.4f}% count: {count55}"
            )
            print(
                f"LogReg accuracy with >=60% confidence: {logreg60*100:.4f}% count: {count60}"
            )
            print(
                f"LogReg accuracy with >=65% confidence: {logreg65*100:.4f}% count: {count65}"
            )
            print()
        if args.train_rbf:
            print("RBF SVM Predictor:")
            # print("Training RBF SVC Predictor...")
            # rbfPredictor.train_rbf(X_train, y_train)
            rbf_probas = rbfPredictor.predict_proba_rbf(X_test)
            count50, rbf50 = get_confidence_accuracy(rbf_probas, y_test, 0.00)
            count55, rbf55 = get_confidence_accuracy(rbf_probas, y_test, 0.05)
            count60, rbf60 = get_confidence_accuracy(rbf_probas, y_test, 0.10)
            count65, rbf65 = get_confidence_accuracy(rbf_probas, y_test, 0.15)
            print(f"RBF accuracy with >=50% confidence: {rbf50*100:.4f}% count: {count50}")
            print(f"RBF accuracy with >=55% confidence: {rbf55*100:.4f}% count: {count55}")
            print(f"RBF accuracy with >=60% confidence: {rbf60*100:.4f}% count: {count60}")
            print(f"RBF accuracy with >=65% confidence: {rbf65*100:.4f}% count: {count65}")
            print()
        if args.train_mlp:
            print("MLP Predictor:")
            # print("Training MLP Predictor...")
            # mlpPredictor.train_mlp(X_train, y_train)
            mlp_probas = mlpPredictor.predict_proba_mlp(X_test)
            count50, mlp50 = get_confidence_accuracy(mlp_probas, y_test, 0.00)
            count55, mlp55 = get_confidence_accuracy(mlp_probas, y_test, 0.05)
            count60, mlp60 = get_confidence_accuracy(mlp_probas, y_test, 0.10)
            count65, mlp65 = get_confidence_accuracy(mlp_probas, y_test, 0.15)
            print(f"MLP accuracy with >=50% confidence: {mlp50*100:.4f}% count: {count50}")
            print(f"MLP accuracy with >=55% confidence: {mlp55*100:.4f}% count: {count55}")
            print(f"MLP accuracy with >=60% confidence: {mlp60*100:.4f}% count: {count60}")
            print(f"MLP accuracy with >=65% confidence: {mlp65*100:.4f}% count: {count65}")
        print("=" * len(header))
        print()

    if args.predict:
        predict_by_id(args.predict)


if __name__ == "__main__":
    main()