import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models import Word2Vec 

import config
import preprocessing
import DotaPredictor
import algPredictor # Needed for the single match prediction

class DotaNeuralNet(nn.Module):
    def __init__(self, num_heroes, embedding_dim, stats_dim):
        super(DotaNeuralNet, self).__init__()
        
        # 1. Embedding Layer
        # Learned from scratch. padding_idx=0 ensures empty slots don't move.
        self.embedding = nn.Embedding(num_heroes, embedding_dim, padding_idx=0)

        # 2. Team Encoder (Siamese Shared Weights)
        # Process each hero vector, then aggregate them into a team vector
        self.hero_processor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 3. Classifier Head
        # Inputs: 
        # - Radiant Team (Sum + Max) = 128
        # - Dire Team (Sum + Max)    = 128
        # - Interaction (Rad - Dire) = 128
        # - Stats (AlgPredictor)     = stats_dim

        # Radiant (128) + Dire (128) + Mult (128) + Diff (128) + Stats (3)
        combined_dim = (128 * 4) + stats_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def process_team(self, hero_indices):
        # hero_indices: [Batch, 5]
        embedded = self.embedding(hero_indices) # [Batch, 5, Emb_Dim]
        
        # Transform latent space
        features = self.hero_processor(embedded) # [Batch, 5, 64]
        
        # Deep Sets Aggregation
        # Sum: Captures total team composition (e.g., total tankiness)
        # Max: Captures specific capabilities (e.g., does anyone have a hex?)
        x_sum = torch.sum(features, dim=1) # [Batch, 64]
        x_max, _ = torch.max(features, dim=1) # [Batch, 64]
        
        return torch.cat([x_sum, x_max], dim=1) # [Batch, 128]

    def forward(self, rad_ids, dire_ids, stats):
        # Siamese Architecture: Pass both teams through the exact same processor
        rad_vec = self.process_team(rad_ids)
        dire_vec = self.process_team(dire_ids)
        
        # Explicit Interaction: Learn the net difference in the latent space
        diff = rad_vec - dire_vec
        prod = rad_vec * dire_vec
        
        # Combine everything
        combined = torch.cat([rad_vec, dire_vec, diff, prod, stats], dim=1)
        
        return self.classifier(combined)

class TorchPredictor:
    def __init__(self):
        self.filename = 'torch_model.pth'
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self):
        # Ensure config.MAX_HERO_ID is set (usually ~138-150)
        # We add a buffer to be safe
        NUM_HEROES = getattr(config, 'MAX_HERO_ID', 155) + 10
        EMB_DIM = 16  
        STATS_DIM = 3 # AlgPredictor score
        
        self.model = DotaNeuralNet(NUM_HEROES, EMB_DIM, STATS_DIM)

        w2v_path = os.path.join(config.MODELS_FOLDER, "embeddings.model")
        if os.path.exists(w2v_path):
            print(f"Loading pre-trained embeddings from {w2v_path}...", file=sys.stderr)
            try:
                w2v_model = Word2Vec.load(w2v_path)
                w2v_weights = w2v_model.wv
            
                # Get the weight matrix from the PyTorch model
                # We wrap it in no_grad to avoid tracking this operation in the graph
                with torch.no_grad():
                    embedding_matrix = self.model.embedding.weight
                
                    hits = 0
                    # Iterate over all possible Hero IDs in the PyTorch layer
                    for i in range(1, NUM_HEROES):
                        # PyTorch uses Int(i), Word2Vec uses Str('i')
                        w2v_key = str(i)
                    
                        if w2v_key in w2v_weights:
                            # Copy the vector
                            embedding_matrix[i] = torch.tensor(w2v_weights[w2v_key])
                            hits += 1
                        
                print(f"Transfer successful: {hits} hero vectors loaded.", file=sys.stderr)
            
            except Exception as e:
                print(f"Failed to load embeddings: {e}", file=sys.stderr)
        else:
            print("No pre-trained embeddings found. Starting from scratch.", file=sys.stderr)

        self.model.to(self.device)

    def train(self, X_train, y_train, epochs=20, batch_size=64):
        if self.model is None:
            self.build_model()
            
        # Expecting X format from preprocessing.generate_pytorch_vector: 
        # [R1, R2, R3, R4, R5, D1, D2, D3, D4, D5, Stat]
        
        X_rad = torch.LongTensor(X_train[:, 0:5]).to(self.device)
        X_dire = torch.LongTensor(X_train[:, 5:10]).to(self.device)
        X_stats = torch.FloatTensor(X_train[:, 10:]).to(self.device)
        
        # Map labels from {-1, 1} to {0, 1} for BCELoss
        y_mapped = (y_train + 1) / 2
        y_tensor = torch.FloatTensor(y_mapped).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_rad, X_dire, X_stats, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        # AdamW is generally better for embeddings than standard Adam
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        print(f"Training PyTorch Model on {self.device}...", file=sys.stderr)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for r_batch, d_batch, s_batch, y_batch in loader:
                if np.random.rand() > 0.5: # mask 50% of batches
                    # Create a mask to zero out some heroes
                    mask_r = torch.bernoulli(torch.full_like(r_batch, 0.7, dtype=torch.float)).long() # Keep 70% of heroes
                    mask_d = torch.bernoulli(torch.full_like(d_batch, 0.7, dtype=torch.float)).long()
                    r_batch = r_batch * mask_r
                    d_batch = d_batch * mask_d
                optimizer.zero_grad()
                outputs = self.model(r_batch, d_batch, s_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}", file=sys.stderr)
            
        self.save_model()

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_rad = torch.LongTensor(X_test[:, 0:5]).to(self.device)
        X_dire = torch.LongTensor(X_test[:, 5:10]).to(self.device)
        X_stats = torch.FloatTensor(X_test[:, 10:]).to(self.device)
        
        with torch.no_grad():
            probas = self.model(X_rad, X_dire, X_stats).cpu().numpy()
            
        # Convert probas back to -1 / 1
        predictions = np.where(probas > 0.5, 1, -1)
        
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        return accuracy, cm

    def predict_proba(self, X):
        self.model.eval()
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_rad = torch.LongTensor(X[:, 0:5]).to(self.device)
        X_dire = torch.LongTensor(X[:, 5:10]).to(self.device)
        X_stats = torch.FloatTensor(X[:, 10:]).to(self.device)
        
        with torch.no_grad():
            pos_prob = self.model(X_rad, X_dire, X_stats).cpu().numpy()
            
        return np.hstack([1 - pos_prob, pos_prob])

    def save_model(self):
        if not os.path.exists(config.MODELS_FOLDER):
            os.makedirs(config.MODELS_FOLDER)
        path = os.path.join(config.MODELS_FOLDER, self.filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved in '{path}'", file=sys.stderr)

    def load_model(self):
        path = os.path.join(config.MODELS_FOLDER, self.filename)
        if os.path.exists(path):
            self.build_model()
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded {self.filename}", file=sys.stderr)
        else:
            print(f"Model file {path} not found", file=sys.stderr)

    def predict_by_match_id(self, match_id):
        # 1. Fetch match data
        match = DotaPredictor.get_match_by_id(match_id)
        if match is None:
             print(f"Match id {match_id} not found in data", file=sys.stderr)
             return None
        
        # 2. Generate vector using the PyTorch-specific generator
        X = preprocessing.generate_pytorch_vector_by_match(match)

        # 3. Predict
        return self.predict_proba(X)[0]