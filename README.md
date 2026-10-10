# Dota 2 Match Outcome Predictor

A Python command-line project that estimates **Radiant's win probability** from the heroes picked by both teams in a Dota 2 match. It combines historical hero matchup statistics from STRATZ with machine-learning models to explore how much team composition can explain match outcomes.

The project includes data collection through a GraphQL API, match filtering, feature engineering, model training and evaluation, saved-model loading, and draft prediction. It is an experimental model-comparison project: it does not observe live gameplay, player skill, items, or match events.

**Technologies:** Python, PyTorch, scikit-learn, CatBoost, Gensim/Word2Vec, NumPy, and STRATZ GraphQL.

## How to use

### 1. Install

Python **3.13** is the version used by the existing development environment. Visual Studio is optional; the program runs from a terminal.

```sh
git clone https://github.com/VadiMIM11/DotaPredictor.git
cd DotaPredictor/DotaPredictor
python -m venv env
```

Activate the environment:

```powershell
# Windows PowerShell
.\env\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source env/bin/activate
```

Then install dependencies and inspect the CLI:

```sh
python -m pip install -r requirements.txt
python DotaPredictor.py --help
```

Run all commands below from the inner **`DotaPredictor/` directory**, alongside `DotaPredictor.py`. Data and model paths are relative to that directory.

### 2. Collect and filter data

Datasets, API credentials, cached features, and trained models are **not included in Git**. A fresh clone needs data collection and training before it can predict.

Obtain a STRATZ API token and save only its value, without a trailing newline, in `data/token`. Create the `data/` directory first. That directory is ignored by Git. Pass the token file with `--stratz`; do not put credentials in `config.py`.

Before fetching, review [config.py](DotaPredictor/config.py). Set `LATEST_MATCH_ID` to the match ID from which you want to scan backward. For an initial run, reduce `FETCH_TRAIN_SIZE`, for example to `5000`; its default is `500000`, which can produce a large download. The setting counts match IDs queried, not valid training examples. `MIN_RANK`, `RANKED_ONLY`, and `GAMEMODES` control filtering.

```sh
python DotaPredictor.py --stratz data/token --update_stats --fetch_train --filter
```

This creates `data/all_hero_stats.json`, `data/raw_train.json`, and `data/clean_train.json`. Fetching requires network access and a working STRATZ token. Keep terminal logs containing token prefixes private.

### 3. Train

Train the PyTorch model used for draft prediction:

```sh
python DotaPredictor.py --train_nn
```

Training creates cached neural-network features in `data/`, saves weights to `models/torch_model.pth`, and prints accuracy and a confusion matrix on a 25% test split. PyTorch uses CUDA when available and otherwise uses the CPU.

To train and compare all implemented ML models instead:

```sh
python DotaPredictor.py --train_all
```

This also generates Word2Vec embeddings and standard-model features. It can take substantially longer because several models use cross-validation.

### 4. Predict a draft

Provide **10 numeric hero IDs**: five Radiant heroes followed by five Dire heroes. Spaces and commas are accepted. Use `0` for an unfilled slot while keeping five slots per team.

```sh
python DotaPredictor.py --load_data --load_nn --predict "11 131 84 16 54, 128 56 82 119 108"
```

Example output from an existing local checkpoint:

```text
Radiant Win Probability: 47.81%
```

The value depends on your data and trained weights. `--predict` uses the PyTorch model. `--load_data` reuses cached features; the current CLI prepares a dataset even when loading a model for prediction, so keep the generated data as well as the weights. The matchup statistics file is also required for inference.

To inspect accuracy at different prediction-confidence thresholds:

```sh
python DotaPredictor.py --load_data --load_nn --test_calibration
```

Despite its name, `--test_calibration` reports accuracy and sample coverage at confidence thresholds; it does not fit a probability calibrator. After changing the dataset or matchup statistics, regenerate features and retrain rather than reusing stale caches.

## How it works

```text
STRATZ GraphQL -> raw matches + hero statistics -> filtered matches
              -> feature engineering -> train/test split -> models
              -> evaluation / saved-model draft prediction
```

1. **Collect and clean matches.** `stratzQueries.py` fetches match outcomes, draft picks, and hero statistics. The CLI filters matches by configured rank, lobby, and game-mode criteria and removes bans from the draft records. Labels are `+1` for a Radiant win and `-1` for a Dire win.
2. **Build matchup features.** `preprocessing.py` converts hero statistics into lookup tables. `algPredictor.py` averages log-odds of Radiant teammate win rates and Radiant-versus-Dire matchup win rates, then combines them into a statistical score. Those two averages and the score form three numeric model inputs.
3. **Represent each draft.** Standard ML models receive the sum of each team's 16-dimensional Word2Vec hero embeddings plus the three matchup features. The PyTorch model receives ten hero IDs plus those features. A separate multi-hot generator encodes Radiant heroes as `+1` and Dire heroes as `-1`, but it is not the representation used by the current standard-model training path.
4. **Train and evaluate.** The CLI uses a 75/25 random train/test split with `RANDOM_STATE = 42`. Models are persisted under `models/`, and training reports test accuracy and confusion matrices. Some standard models additionally use isotonic probability calibration with cross-validation.
5. **Estimate a new draft.** The PyTorch model embeds each hero, passes both teams through the same encoder, and aggregates each team's hero features with sum and maximum pooling. This Deep Sets-style aggregation makes the team representation independent of hero order at inference. The classifier combines both team vectors, their difference and elementwise product, and the matchup features, then outputs a sigmoid probability for a Radiant win.

The neural network can initialize its hero embeddings from `models/embeddings.model` when present; otherwise it learns them from scratch. Training also randomly masks hero IDs in some batches to expose the network to incomplete drafts.

### Models

| Model | Implementation |
| --- | --- |
| Matchup-statistics baseline | `algPredictor.py` |
| Logistic regression | `logRegPredictor.py` |
| RBF support-vector machine | `rbfPredictor.py` |
| Multilayer perceptron | `mlpPredictor.py` |
| Random forest | `treePredictor.py` |
| CatBoost classifier | `catPredictor.py` |
| PyTorch network with a shared team encoder | `torchPredictor.py` |

The CLI calls the random-forest option `--train_tree` / `--load_tree`; it is not a single decision tree. Individual ML models also have corresponding training/loading flags, such as `--train_logreg` and `--load_logreg`.

## Current limitations

- Predictions use hero composition and aggregate matchup statistics. They do not establish a causal effect of a pick or guarantee a match result. Partial-draft estimates are experimental.
- Evaluation is exploratory: standard features are scaled before the test split, and fetched matchup statistics are not tied to each historical match's date. A time-separated evaluation with preprocessing fitted only on training data would be needed for stronger performance claims.
- Hero-ID bounds, the fetch starting point, and match filters are configured manually. Review them when using a different dataset or game patch.
- The statistical baseline is used in feature engineering, but its standalone `--test_alg_predict` path needs repair. Standard-model `--predict_by_id` paths also use a different feature representation from training. The usage examples above use PyTorch draft prediction.
- Hero-pick recommendation is unfinished and is not a supported workflow. Any local `heroPicker.py` / `--suggest_pick` work should be treated as a draft feature.
