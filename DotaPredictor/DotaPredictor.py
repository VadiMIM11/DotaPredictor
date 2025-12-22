import argparse
import sys
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
from logRegPredictor import LogRegPredictor
from rbfPredictor import RbfPredictor
from mlpPredictor import MlpPredictor
from treePredictor import TreePredictor

logRegPredictor = LogRegPredictor()
rbfPredictor = RbfPredictor()
mlpPredictor = MlpPredictor()
treePredictor = TreePredictor()


def extract_hero_ids(feature_vector):
    radiant_heroes = [i for i, val in enumerate(feature_vector) if val == 1]
    dire_heroes = [i for i, val in enumerate(feature_vector) if val == -1]
    return radiant_heroes, dire_heroes


def generate_feature_vector(match_json):
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
    for key, match in matches.items():
        feature_vector = generate_feature_vector(match)
        label = get_label(match)
        X_list.append(feature_vector)
        y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def update_local_stats(api_token):
    print(
        "Updating local hero statistics dataset via stratz.com requests...",
        file=sys.stderr,
    )
    stats = sq.fetch_all_stats(api_token)

    FOLDER = config.DATA_FOLDER
    FILE_NAME = "all_hero_stats.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}", file=sys.stderr)

    try:
        with open(PATH, "w", encoding=config.DEFAULT_ENCODING) as f:
            json.dump(stats, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}", file=sys.stderr)

    num_fetched = len(stats.get("data", {}).get("heroStats", {}))
    print(f"Fetched stats for {num_fetched} heroes", file=sys.stderr)
    if num_fetched != config.NUM_HEROES:
        print(
            f"Warning: Expected {config.NUM_HEROES} heroes, but got {num_fetched}.",
            file=sys.stderr,
        )

    print(f"Local hero stats dataset updated and saved to {PATH}", file=sys.stderr)


def get_match_by_id(match_id):
    try:
        with open(
            f"{config.DATA_FOLDER}/clean_train.json",
            "r",
            encoding=config.DEFAULT_ENCODING,
        ) as f:
            matches = json.load(f)
        matches = matches.get("data")
        if matches is None:
            print("Error reading data file", file=sys.stderr)
            return None
        for key, match in matches.items():
            current_id = match.get("id")
            if not current_id is None and current_id == match_id:
                return match
    except IOError:
        print(
            "Data file not found. Please run the data fetching first.", file=sys.stderr
        )
        exit(1)

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
        print(f"Created folder: {FOLDER}", file=sys.stderr)
    try:
        with open(PATH, "w", encoding=config.DEFAULT_ENCODING) as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}", file=sys.stderr)

    print(f"Raw training dataset saved to {PATH}", file=sys.stderr)


def clean_train():
    try:
        with open(
            os.path.join(config.DATA_FOLDER, "raw_train.json"),
            encoding=config.DEFAULT_ENCODING,
        ) as f:
            data = json.load(f)
    except IOError as e:
        print(f"Could not access raw training data file: {e}", file=sys.stderr)
        exit(1)

    FOLDER = config.DATA_FOLDER
    FILE_NAME = "clean_train.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}", file=sys.stderr)

    matches = data["data"]
    print("Raw train size:", len(matches), file=sys.stderr)
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
    print("Clean train size:", len(filtered_matches), file=sys.stderr)
    data["data"] = filtered_matches

    try:
        with open(PATH, "w", encoding=config.DEFAULT_ENCODING) as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving file: {e}", file=sys.stderr)

    print(f"Clean training dataset saved to {PATH}", file=sys.stderr)


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


def main():
    parser = argparse.ArgumentParser(
        description="Welcome to the set of statistical and ML models to predict DOTA 2 match outcome"
    )

    parser.add_argument(
        "-u",
        "--update_stats",
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
        "--load_all",
        action="store_true",
        required=False,
        help="Load all models (Logistic Regression, RBF SVM, MLP, Decision Tree) from disk",
    )

    parser.add_argument(
        "--load_logreg",
        action="store_true",
        required=False,
        help="Load logistic regression model from disk",
    )

    parser.add_argument(
        "--load_rbf",
        action="store_true",
        required=False,
        help="Load RBF SVM model from disk",
    )

    parser.add_argument(
        "--load_mlp",
        action="store_true",
        required=False,
        help="Load MLP model from disk",
    )

    parser.add_argument(
        "--load_tree",
        action="store_true",
        required=False,
        help="Load Decision Tree model from disk",
    )

    parser.add_argument(
        "--train_all",
        action="store_true",
        required=False,
        help="Train all models (Logistic Regression, RBF SVM, MLP, Decision Tree) on training dataset",
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
        "--train_mlp",
        action="store_true",
        required=False,
        help="Train MLP model on training dataset",
    )

    parser.add_argument(
        "--train_tree",
        action="store_true",
        required=False,
        help="Train Decision Tree model on training dataset",
    )

    parser.add_argument(
        "--predict_by_id",
        type=int,
        required=False,
        help="Predict match outcome by match ID using all models",
    )

    parser.add_argument(
        "--test_calibration",
        action="store_true",
        required=False,
        help="Test calibration of models on test dataset",
    )

    args = parser.parse_args()

    if (
        args.load_all
        or args.load_logreg
        or args.load_rbf
        or args.load_mlp
        or args.load_tree
    ):
        if args.load_all or args.load_logreg:
            logRegPredictor.load_model()

        if args.load_all or args.load_rbf:
            rbfPredictor.load_model()

        if args.load_all or args.load_mlp:
            mlpPredictor.load_model()

        if args.load_all or args.load_tree:
            treePredictor.load_model()

    if args.stratz:
        print("Stratz API token file path:", args.stratz, file=sys.stderr)
        token = ""
        try:
            with open(args.stratz, "r", encoding=config.DEFAULT_ENCODING) as f:
                token = f.read()
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            exit(1)
        print("Stratz API Token: ", token[:10] + "...", file=sys.stderr)
        config.API_TOKEN = token

    if args.update_stats:
        if not args.stratz:
            print(
                "Please provide stratz.com API token to update local hero stats dataset using:\n --stratz <PATH>",
                file=sys.stderr,
            )
            return

        update_local_stats(config.API_TOKEN)

    if args.fetch_train:
        if not args.stratz:
            print(
                "Please provide stratz.com API token to fetch latest matches for training dataset using:\n --stratz <PATH>",
                file=sys.stderr,
            )
            return
        json_data = sq.fetch_train(config.API_TOKEN)
        save_raw_train(json_data)

    if args.filter:
        clean_train()

    global X_train, X_test, y_train, y_test
    X_train = X_test = y_train = y_test = None

    if (
        args.load_all
        or args.load_rbf
        or args.load_logreg
        or args.load_mlp
        or args.load_tree
    ) or (
        args.train_all
        or args.train_rbf
        or args.train_logreg
        or args.train_mlp
        or args.train_tree
    ):
        try:
            with open(
                os.path.join(config.DATA_FOLDER, "clean_train.json"),
                encoding=config.DEFAULT_ENCODING,
            ) as f:
                data = json.load(f)
        except:
            print(
                "Could not access clean training data file. Please run with --filter",
                file=sys.stderr,
            )
            return

        X, y = generate_treining_set(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=config.RANDOM_STATE
        )

    if args.train_all or args.train_tree:
        treePredictor.train(X_train, y_train)
        print(file=sys.stderr)
        header = "=" * 5 + " Decision Tree Predictor " + "=" * 5
        print(header, file=sys.stderr)
        print("Total matches to test:", y_test.size, file=sys.stderr)
        print(file=sys.stderr)
        accuracy, cm = treePredictor.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)
        precision = (
            cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        )  # Precision TP / (TP + FP)
        sensitivity = (
            cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        )  # Sensitivity (Recall) TP / (TP + FN)
        specificity = (
            cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        )  # Specificity TN / (TN + FP)
        print(f"Precision: {precision:.4f}", file=sys.stderr)
        print(f"Sensitivity: {sensitivity:.4f}", file=sys.stderr)
        print(f"Specificity: {specificity:.4f}", file=sys.stderr)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    if args.train_all or args.train_logreg:
        logRegPredictor.train(X_train, y_train)
        print(file=sys.stderr)
        header = "=" * 5 + " Logistic Regression Predictor " + "=" * 5
        print(header, file=sys.stderr)
        print("Total matches to test:", y_test.size, file=sys.stderr)
        print(file=sys.stderr)
        accuracy, cm = logRegPredictor.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)
        # print("True Negatives:", cm[0][0])
        # print("False Positives:", cm[0][1])
        # print("False Negatives:", cm[1][0])
        # print("True Positives:", cm[1][1])
        precision = (
            cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        )  # Precision TP / (TP + FP)
        sensitivity = (
            cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        )  # Sensitivity (Recall) TP / (TP + FN)
        specificity = (
            cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        )  # Specificity TN / (TN + FP)
        print(f"Precision: {precision:.4f}", file=sys.stderr)
        print(f"Sensitivity: {sensitivity:.4f}", file=sys.stderr)
        print(f"Specificity: {specificity:.4f}", file=sys.stderr)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    if args.train_all or args.train_rbf:
        rbfPredictor.train(X_train, y_train)
        print(file=sys.stderr)
        header = "=" * 5 + " RBF SVM Predictor " + "=" * 5
        print(header, file=sys.stderr)
        print("Total matches to test:", y_test.size, file=sys.stderr)
        print(file=sys.stderr)
        accuracy, cm = rbfPredictor.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)
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
        print(f"Precision: {precision:.4f}", file=sys.stderr)
        print(f"Sensitivity: {sensitivity:.4f}", file=sys.stderr)
        print(f"Specificity: {specificity:.4f}", file=sys.stderr)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    if args.train_all or args.train_mlp:
        mlpPredictor.train(X_train, y_train)
        print(file=sys.stderr)
        header = "=" * 5 + " MLP Predictor " + "=" * 5
        print(header, file=sys.stderr)
        print("Total matches to test:", y_test.size, file=sys.stderr)
        print(file=sys.stderr)
        accuracy, cm = mlpPredictor.evaluate(X_test, y_test, file=sys.stderr)
        print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)
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
        print(f"Precision: {precision:.4f}", file=sys.stderr)
        print(f"Sensitivity: {sensitivity:.4f}", file=sys.stderr)
        print(f"Specificity: {specificity:.4f}", file=sys.stderr)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    if args.test_alg_predict:
        from algPredictor import predict

        try:
            with open(
                os.path.join(config.DATA_FOLDER, "clean_train.json"),
                encoding=config.DEFAULT_ENCODING,
            ) as f:
                data = json.load(f)
        except IOError as e:
            print(f"Could not access clean training data file: {e}", file=sys.stderr)
            exit(1)
        try:
            with open(
                os.path.join("data", "all_hero_stats.json"),
                "r",
                encoding=config.DEFAULT_ENCODING,
            ) as f:
                all_stats = json.load(f)
        except IOError as e:
            print(f"Could not access hero stats file: {e}", file=sys.stderr)
            print("Try fetching hero stats from stratz using --update", file=sys.stderr)
            exit(1)

        print(file=sys.stderr)
        header = "=" * 5 + " Statistical Prediictor " + "=" * 5
        print(header, file=sys.stderr)
        matches = data["data"]
        total_matches = len(matches)
        print("Total matches to test:", total_matches, file=sys.stderr)
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
                    print(
                        f"Match {match_key} has no radiantWin field, skipping...",
                        file=sys.stderr,
                    )
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
                print(f"Error processing match {match_key}: {e}", file=sys.stderr)
                continue
        print(file=sys.stderr)
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
        print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)
        print(f"Precision: {precision:.4f}", file=sys.stderr)
        print(f"Sensitivity: {sensitivity:.4f}", file=sys.stderr)
        print(f"Specificity: {specificity:.4f}", file=sys.stderr)

        # print("HIghly Outdrafted Matches:", outdrafted)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    def print_confidence_accuracy(confidence, accuracy, count):
        print(
            f"Confidence >={confidence*100:.1f}%: Accuracy: {accuracy*100:.4f}% count: {count}",
            file=sys.stderr,
        )

    if args.test_calibration:
        header = "=" * 5 + " Calibration Test " + "=" * 5
        print(header, file=sys.stderr)
        print(file=sys.stderr)

        thresholds = [0.00, 0.05, 0.10, 0.15]
        probas = []

        if args.test_alg_predict:
            print("Statistical Predictor:", file=sys.stderr)
            try:
                with open(
                    os.path.join("data", "all_hero_stats.json"),
                    "r",
                    encoding=config.DEFAULT_ENCODING,
                ) as f:
                    all_stats = json.load(f)
            except IOError as e:
                print(f"Could not access hero stats file: {e}", file=sys.stderr)
                print("Try fetching hero stats from stratz using --update", file=sys.stderr)
                exit(1)

            for i in range(len(X_test)):
                prob = algPredictor.predict(X_test[i], all_stats)
                probas.append([1 - prob, prob])

            for threshold in thresholds:
                count, accuracy = get_confidence_accuracy(probas, y_test, threshold)
                print_confidence_accuracy(0.5 + threshold, accuracy, count)
            print(file=sys.stderr)

        if (args.load_all or args.load_logreg) or (args.train_all or args.train_logreg):
            print("Logistic Regression Predictor:", file=sys.stderr)
            probas = logRegPredictor.predict_proba(X_test)
            for threshold in thresholds:
                count, accuracy = get_confidence_accuracy(probas, y_test, threshold)
                print_confidence_accuracy(0.5 + threshold, accuracy, count)
            print(file=sys.stderr)
        if (args.load_all or args.load_rbf) or (args.train_all or args.train_rbf):
            print("RBF SVM Predictor:", file=sys.stderr)
            probas = rbfPredictor.predict_proba(X_test)
            for threshold in thresholds:
                count, accuracy = get_confidence_accuracy(probas, y_test, threshold)
                print_confidence_accuracy(0.5 + threshold, accuracy, count)
            print(file=sys.stderr)
        if (args.load_all or args.load_mlp) or (args.train_all or args.train_mlp):
            print("MLP Predictor:", file=sys.stderr)
            probas = mlpPredictor.predict_proba(X_test)
            for threshold in thresholds:
                count, accuracy = get_confidence_accuracy(probas, y_test, threshold)
                print_confidence_accuracy(0.5 + threshold, accuracy, count)
            print(file=sys.stderr)

        if (args.load_all or args.load_tree) or (args.train_all or args.train_tree):
            print("Decision Tree Predictor:", file=sys.stderr)
            probas = treePredictor.predict_proba(X_test)
            for threshold in thresholds:
                count, accuracy = get_confidence_accuracy(probas, y_test, threshold)
                print_confidence_accuracy(0.5 + threshold, accuracy, count)
            print(file=sys.stderr)
        print("=" * len(header), file=sys.stderr)
        print(file=sys.stderr)

    def predict_by_id(m_id):
        # m_id = 8525383837
        print(f"Precict match {m_id}", file=sys.stderr)
        if (args.load_all or args.load_rbf) or (args.train_all or args.train_rbf):
            prob = rbfPredictor.predict_by_match_id(m_id)
            print(
                f"RBF: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%",
                file=sys.stderr,
            )
            print(f"{rbfPredictor.filename}: {prob[1]}", file=sys.stdout)
        if (args.load_all or args.load_mlp) or (args.train_all or args.train_mlp):
            prob = mlpPredictor.predict_by_match_id(m_id)
            print(
                f"MLP: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%",
                file=sys.stderr,
            )
            print(f"{mlpPredictor.filename}: {prob[1]}", file=sys.stdout)
        if (args.load_all or args.load_tree) or (args.train_all or args.train_tree):
            prob = treePredictor.predict_by_match_id(m_id)
            print(
                f"Tree: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%",
                file=sys.stderr,
            )
            print(f"{treePredictor.filename}: {prob[1]}", file=sys.stdout)
        if (args.load_all or args.load_logreg) or (args.train_all or args.train_logreg):
            prob = logRegPredictor.predict_by_match_id(m_id)
            print(
                f"LogReg: {m_id} -> Radiant Win with {(prob[1] * 100):.4f}%",
                file=sys.stderr,
            )
            print(f"{logRegPredictor.filename}: {prob[1]}", file=sys.stdout)
        prob = algPredictor.predict_by_match_id(m_id)
        print(
            f"Statistical: {m_id} -> Radiant Win with {(prob * 100):.4f}%",
            file=sys.stderr,
        )
        # print(f"Statistical: {prob}", file=sys.stdout)
        print(file=sys.stderr)

    if args.predict_by_id:
        predict_by_id(args.predict)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise

    print("Press any key to exit...", file=sys.stderr)
    input()