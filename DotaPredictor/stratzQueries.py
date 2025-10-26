from ast import Raise
import requests
import json
import os
import copy
from tqdm import tqdm

import config

GET_HERO_STATS_QUERY = """
query getHeroStats($id: Short!) {
  heroStats{
    matchUp(
      heroId: $id
      take: 200
    ){
      heroId
      with {
        heroId2
        winsAverage
      }
      vs{
        heroId2
        winsAverage
      }
    }
  }
}
"""


URL = "https://api.stratz.com/graphql"



def generate_hero_ids(api_token):
    print("Obtaining ids of all existing heroes from single stratz query...")
    hero_stats = fetch_hero_stats(1, api_token)
    arr = hero_stats["data"]["heroStats"]["matchUp"][0]["with"]
    all_ids = [1]
    for entry in arr:
        h_id = entry["heroId2"]
        if not h_id in all_ids:
            all_ids.append(h_id)

    FOLDER = config.DATA_FOLDER
    FILE_NAME = "hero_ids.json"
    PATH = os.path.join(FOLDER, FILE_NAME)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}")
    try:
        with open(PATH, "w") as f:
            json.dump(all_ids, f)
    except IOError as e:
        print(f"Error saving file: {e}")

    print(f"Success! Ids saved to {PATH}")


def generate_fetch_all_query(api_token):
    IDS_PATH = os.path.join(config.DATA_FOLDER, "hero_ids.json")
    all_ids = []
    query = ""
    try:
        with open(IDS_PATH, "r") as f:
            all_ids = json.load(f)
        print("Success! Hero ids obtained from a local file!")
    except IOError as e:
        print(f"Error reading file: {e}")
        generate_hero_ids(api_token)
        try:
            with open(IDS_PATH, "r") as f:
                all_ids = json.load(f)
        except IOError as e:
            print(f"Error reading file: {e}")
            exit(1)

    # print("Length of all ids: ", len(all_ids))

    query = "query MultipleHeroMatchups {\n  heroStats {\n"
    for i, h_id in enumerate(all_ids):
        query += f"    # {i+1}. Alias the execution (for Hero {h_id})\n"
        query += f"    hero{h_id}MatchUp: matchUp(\n"
        query += f"      heroId: {h_id}\n"
        query += f"      take: 200\n"
        query += f"    ) {{\n"
        query += f"      heroId\n"
        query += f"      with {{\n"
        query += f"        heroId2\n"
        query += f"        winsAverage\n"
        query += f"      }}\n"
        query += f"      vs {{\n"
        query += f"        heroId2\n"
        query += f"        winsAverage\n"
        query += f"      }}\n"
        query += f"    }}\n\n"

    query += "  }\n}"
    return query



def generate_fetch_train_query(api_token, latest_match_id, fetch_size):
    if fetch_size < 1:
        Raise("fetch_size must be at least 1")

    query = "query getTrainingSet {\n"
    for i in range(fetch_size):
        match_id = latest_match_id - i
        query += f"match{match_id}: match(id: {match_id})"
        query += "{\n"
        query += "id\n"
        query += "didRadiantWin\n"
        query += "lobbyType\n"
        query += "gameMode\n"
        query += "bracket\n"
        query += "pickBans {\n"
        query += "isPick\n"
        query += "heroId\n"
        query += "bannedHeroId\n"
        query += "isRadiant\n"
        query += "}\n"
        query += "}\n\n"
    query += "}"

    # print("Generated training set query:\n", query)
    return query


def combine_query_response(response1, response2):
    combined_response = copy.deepcopy(response1)

    data1 = combined_response.get("data")
    data2 = response2.get("data")
    # print(type(data1), type(data2))
    if isinstance(data1, dict) and isinstance(data2, dict):
        data1.update(data2)
    elif data1 is None and isinstance(data2, dict):
        combined_response["data"] = copy.deepcopy(data2)

    return combined_response

def fetch_all_winrates(api_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "STRATZ_API",
    }
    query = """
        query fetchAllWinrates {
          heroStats{
            stats{
              heroId
              matchCount
              winCount
            }
          }
        }
        """
    print("Fetching all hero winrates from stratz...")
    response = requests.post( 
        URL, json={"query": query}, headers=headers
    )
    if response.status_code == 200:
        print("Success! Fetched winrates for all heroes!")

        return response.json()
    else:
        print("Query failed with code:", response.status_code)
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )

def fetch_train(api_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "STRATZ_API",
    }
    current_latest_id = config.LATEST_MATCH_ID
    current_json = {}
    print("Fetching matches from stratz ...")
    for i in tqdm(range(config.FETCH_TRAIN_SIZE // config.MAX_MATCHES_IN_QUERY)):
        FETCH_TRAIN_QUERY = generate_fetch_train_query(
            api_token, current_latest_id, config.MAX_MATCHES_IN_QUERY
        )
        # print(
        #     f"Matches [{current_latest_id} .. {current_latest_id - config.MAX_MATCHES_IN_QUERY + 1}]"
        # )
        # print(
        #     f"Fetching a batch of {config.MAX_MATCHES_IN_QUERY} matches from stratz..."
        # )

        response = requests.post(
            URL, json={"query": FETCH_TRAIN_QUERY}, headers=headers
        )

        if response.status_code == 200:
            #print("Success! Fetched training set!")
            pass
        else:
            print("Query failed with code:", response.status_code)
            # print("Skipping...")
            raise Exception(
                f"Query failed with status code {response.status_code}: {response.text}"
            )
        current_latest_id -= config.MAX_MATCHES_IN_QUERY
        current_json = combine_query_response(current_json, response.json())

    batch_size = config.FETCH_TRAIN_SIZE % config.MAX_MATCHES_IN_QUERY
    if batch_size > 0:
        FETCH_TRAIN_QUERY = generate_fetch_train_query(
            api_token, current_latest_id, batch_size
        )
        print(f"Matches [{current_latest_id} .. {current_latest_id - batch_size + 1}]")
        print(f"Fetching a batch of {batch_size} matches from stratz...")
        response = requests.post(
            URL, json={"query": FETCH_TRAIN_QUERY}, headers=headers
        )
        if response.status_code == 200:
            print("Success! Fetched training set!")
        else:
            print("Query failed with code:", response.status_code)
            raise Exception(
                f"Query failed with status code {response.status_code}: {response.text}"
            )
        current_json = combine_query_response(current_json, response.json())

    return current_json


def fetch_hero_stats(hero_id, api_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "STRATZ_API",
    }
    variables = {"id": hero_id}

    print(f"Fetching hero stats for hero id {hero_id} from stratz...")
    response = requests.post(
        URL,
        json={"query": GET_HERO_STATS_QUERY, "variables": variables},
        headers=headers,
    )

    if response.status_code == 200:
        print("Success! Fetched hero stats!")
        # print(response.json())
        all_hero_winrates = fetch_all_winrates(api_token)
        data = response.json()
        for stat in all_hero_winrates["data"]["heroStats"]["stats"]:
            heroId = stat["heroId"]
            matchCount = stat["matchCount"]
            winCount = stat["winCount"]
            winRate = winCount / matchCount if matchCount > 0 else 0.0
            heroMatchUp = data["data"]["heroStats"].get(f"hero{heroId}MatchUp")
            if not heroMatchUp is None:
                heroMatchUp[0]["totalMatches"] = matchCount
                heroMatchUp[0]["totalWins"] = winCount
                heroMatchUp[0]["winRate"] = winRate
        return response.json()
    else:
        print("Query failed with code:", response.status_code)
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )


def fetch_all_stats(api_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "STRATZ_API",
    }
    FETCH_ALL_STATS_EQURY = generate_fetch_all_query(api_token)
    # print(FETCH_ALL_STATS_EQURY)
    # exit(1)
    print("Fetching all hero stats from stratz...")
    response = requests.post(
        URL, json={"query": FETCH_ALL_STATS_EQURY}, headers=headers
    )
    if response.status_code == 200:
        print("Success! Fetched stats for all heroes!")
        all_hero_winrates = fetch_all_winrates(api_token)
        data = response.json()
        for stat in all_hero_winrates["data"]["heroStats"]["stats"]:
            heroId = stat["heroId"]
            matchCount = stat["matchCount"]
            winCount = stat["winCount"]
            winRate = winCount / matchCount if matchCount > 0 else 0.0
            heroMatchUp = data["data"]["heroStats"].get(f"hero{heroId}MatchUp")
            if not heroMatchUp is None:
                heroMatchUp[0]["totalMatches"] = matchCount
                heroMatchUp[0]["totalWins"] = winCount
                heroMatchUp[0]["winRate"] = winRate

        return data
    else:
        print("Query failed with code:", response.status_code)
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )
