import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import math

# === 1. Load CSV and parse list columns ===
df = pd.read_csv("game_dataset_cleaned.csv")

print("Loaded game IDs:", df['id'].tolist()[:10], "...")
print("Total games loaded:", len(df))

def parse_list(cell):
    if pd.isna(cell) or cell == "":
        return []
    try:
        return ast.literal_eval(cell)
    except:
        if isinstance(cell, str):
            return [x.strip() for x in cell.split(",") if x.strip()]
        return []

list_cols = ["genres", "themes", "keywords", "game_modes", "player_perspectives", "similar_games"]
for col in list_cols:
    df[col] = df[col].apply(parse_list)

print("Dataframe loaded. Shape:", df.shape)

# === 2. Build tag vocabularies (index maps) for each tag column ===
tag_columns = ["genres", "themes", "keywords", "game_modes", "player_perspectives"]
tag_vocab = {}
tag_to_idx = {}

for col in tag_columns:
    all_tags = set()
    for tags in df[col]:
        all_tags.update(tags)
    vocab = sorted(all_tags)
    tag_vocab[col] = vocab
    tag_to_idx[col] = {tag: idx for idx, tag in enumerate(vocab)}
    print(f"{col} vocab size: {len(vocab)}")

# === 3. Compute IDF weights per tag for weighted averaging ===
def compute_idf_weights(df, tag_columns, tag_vocab):
    idf_weights = {}
    N = len(df)
    for col in tag_columns:
        tag_doc_counts = {tag: 0 for tag in tag_vocab[col]}
        for tags in df[col]:
            unique_tags = set(tags)
            for tag in unique_tags:
                if tag in tag_doc_counts:
                    tag_doc_counts[tag] += 1
        idf = {tag: math.log(N / (1 + count)) for tag, count in tag_doc_counts.items()}
        idf_weights[col] = idf
    return idf_weights

idf_weights = compute_idf_weights(df, tag_columns, tag_vocab)

# === 4. Encode tags as indices for each game, pad to fixed length ===
max_tags_per_cat = {
    col: max(df[col].apply(len).max(), 1) for col in tag_columns
}

def encode_tags(tags, mapping, max_len):
    idxs = [mapping.get(t, -1) for t in tags if t in mapping]
    idxs = [i for i in idxs if i >= 0]
    if len(idxs) < max_len:
        idxs = idxs + [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

def encode_weights(tags, idf_map, max_len):
    weights = [idf_map.get(t, 0.0) for t in tags if t in idf_map]
    if len(weights) < max_len:
        weights = weights + [0.0] * (max_len - len(weights))
    else:
        weights = weights[:max_len]
    return weights

encoded_tag_indices = {}
encoded_tag_weights = {}

for col in tag_columns:
    max_len = max_tags_per_cat[col]
    mapping = tag_to_idx[col]
    idf_map = idf_weights[col]
    encoded_tag_indices[col] = df[col].apply(lambda tags: encode_tags(tags, mapping, max_len)).tolist()
    encoded_tag_weights[col] = df[col].apply(lambda tags: encode_weights(tags, idf_map, max_len)).tolist()

# === 5. Create PyTorch Dataset ===
class TagEmbeddingDataset(Dataset):
    def __init__(self, encoded_tags, encoded_weights, tag_columns):
        self.encoded_tags = encoded_tags
        self.encoded_weights = encoded_weights
        self.tag_columns = tag_columns
        self.length = len(next(iter(encoded_tags.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        for col in self.tag_columns:
            sample[col] = torch.tensor(self.encoded_tags[col][idx], dtype=torch.long)
            sample[col + "_weights"] = torch.tensor(self.encoded_weights[col][idx], dtype=torch.float)
        return sample

dataset = TagEmbeddingDataset(encoded_tag_indices, encoded_tag_weights, tag_columns)

batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === 6. Define model with weighted average in forward ===
class TagEmbeddingAutoencoder(nn.Module):
    def __init__(self, tag_vocab_sizes, max_tags_per_cat, embedding_dim=32, encoding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tag_columns = list(tag_vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embedding_dim)
            for col, vocab_size in tag_vocab_sizes.items()
        })
        self.max_tags_per_cat = max_tags_per_cat
        input_dim = embedding_dim * len(tag_vocab_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        embeddings_per_cat = []
        for col in self.tag_columns:
            emb = self.embeddings[col](inputs[col])  # (batch, max_tags, emb_dim)
            weights = inputs[col + "_weights"].unsqueeze(-1)  # (batch, max_tags, 1)
            weighted_emb = emb * weights
            weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
            emb_weighted_avg = weighted_emb.sum(dim=1) / weight_sum.squeeze(1)
            embeddings_per_cat.append(emb_weighted_avg)
        combined = torch.cat(embeddings_per_cat, dim=1)
        encoded = self.encoder(combined)
        decoded = self.decoder(encoded)
        return decoded, encoded

tag_vocab_sizes = {col: len(vocab) for col, vocab in tag_vocab.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TagEmbeddingAutoencoder(tag_vocab_sizes, max_tags_per_cat, embedding_dim=32, encoding_dim=128).to(device)

# === 7. Training function ===
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 15
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        total_batches = len(dataloader)
        for batch_idx, batch_samples in enumerate(dataloader, start=1):
            inputs = {col: batch_samples[col].to(device) for col in tag_columns}
            for col in tag_columns:
                inputs[col + "_weights"] = batch_samples[col + "_weights"].to(device)

            optimizer.zero_grad()
            decoded, encoded = model(inputs)

            with torch.no_grad():
                embeddings_per_cat = []
                for col in tag_columns:
                    emb = model.embeddings[col](inputs[col])
                    weights = inputs[col + "_weights"].unsqueeze(-1)
                    weighted_emb = emb * weights
                    weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8
                    emb_weighted_avg = weighted_emb.sum(dim=1) / weight_sum.squeeze(1)
                    embeddings_per_cat.append(emb_weighted_avg)
                target = torch.cat(embeddings_per_cat, dim=1)

            loss = criterion(decoded, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs[tag_columns[0]].size(0)

            progress_pct = (batch_idx / total_batches) * 100
            print(f"\rEpoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) - Loss: {loss.item():.4f}", end="")

        avg_loss = epoch_loss / len(dataset)
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save after training
    torch.save(model.state_dict(), "tag_embedding_autoencoder.pth")
    with open("tag_vocab.pkl", "wb") as f:
        pickle.dump({
            "tag_vocab": tag_vocab,
            "tag_to_idx": tag_to_idx,
            "max_tags_per_cat": max_tags_per_cat,
            "tag_columns": tag_columns
        }, f)
    print("Training finished and model saved.")

    # Create embeddings for all games and save
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            inputs = {col: sample[col].unsqueeze(0).to(device) for col in tag_columns}
            for col in tag_columns:
                inputs[col + "_weights"] = sample[col + "_weights"].unsqueeze(0).to(device)
            _, encoded = model(inputs)
            all_embeddings.append(encoded.cpu().numpy()[0])
    all_embeddings_np = np.vstack(all_embeddings)

    # Map game IDs to dataset indices
    id_to_index = {game_id: idx for idx, game_id in enumerate(df["id"])}

    # Map dataset indices to similar games' indices safely
    game_idx_to_similars = {}
    for idx, sim_ids in enumerate(df["similar_games"]):
        if isinstance(sim_ids, list):
            sim_indices = [id_to_index[sid] for sid in sim_ids if sid in id_to_index]
        else:
            sim_indices = []
        game_idx_to_similars[idx] = sim_indices

    np.save("game_embeddings.npy", all_embeddings_np)
    with open("game_idx_to_similars.pkl", "wb") as f:
        pickle.dump(game_idx_to_similars, f)

    print("All game embeddings and similar game mappings saved.")

# === 8. Loading & recommendation part ===
def load_model_and_data():
    global model, tag_vocab, tag_to_idx, max_tags_per_cat, tag_columns
    global all_embeddings_np, game_idx_to_similars, id_to_index, index_to_id

    # Load vocabs
    with open("tag_vocab.pkl", "rb") as f:
        vocabs = pickle.load(f)
        tag_vocab = vocabs["tag_vocab"]
        tag_to_idx = vocabs["tag_to_idx"]
        max_tags_per_cat = vocabs["max_tags_per_cat"]
        tag_columns = vocabs["tag_columns"]

    # Create model again with loaded vocabs
    tag_vocab_sizes = {col: len(vocab) for col, vocab in tag_vocab.items()}
    model = TagEmbeddingAutoencoder(tag_vocab_sizes, max_tags_per_cat, embedding_dim=32, encoding_dim=128).to(device)
    model.load_state_dict(torch.load("tag_embedding_autoencoder.pth", map_location=device))
    model.eval()

    # Load embeddings and similar game mapping
    all_embeddings_np = np.load("game_embeddings.npy")
    with open("game_idx_to_similars.pkl", "rb") as f:
        game_idx_to_similars = pickle.load(f)

    # Build id to index maps from df (ensure consistent with saved)
    id_to_index = {game_id: idx for idx, game_id in enumerate(df["id"])}
    print("First 10 df['id'] values:", df["id"].head(10).tolist())
    print("Type of first ID:", type(df["id"].iloc[0]))
    print("Does 4754 exist in df['id']?:", 4754 in df["id"].values)
    print("Does 4754 exist in id_to_index?:", 4754 in id_to_index)
    index_to_id = {idx: game_id for game_id, idx in id_to_index.items()}

    print("Model, vocabularies, embeddings, and similar games loaded.")

# === 9. Recommendation helper ===
def cosine_sim(a, matrix):
    a_norm = a / np.linalg.norm(a)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm.dot(a_norm)

def recommend_games(input_game_ids, top_k=10, filters=None, base_bonus=0.2, overlap_bonus_pct=0.05, proximity_bonus=0.05):
    input_game_ids = [int(gid) for gid in input_game_ids]

    # Validate input game IDs
    for input_game_id in input_game_ids:
        if input_game_id not in id_to_index:
            print(f"Game ID {input_game_id} not found in dataset.")
            return []

    input_indices = [id_to_index[gid] for gid in input_game_ids]
    input_embeddings = [all_embeddings_np[idx] for idx in input_indices]
    combined_embedding = np.mean(input_embeddings, axis=0)  

    sims = cosine_sim(combined_embedding, all_embeddings_np)

    # Count how many input games each game appears as "similar" to
    similarity_counts = {}
    for idx in input_indices:
        sim_indices = game_idx_to_similars.get(idx, [])
        for sim_idx in sim_indices:
            similarity_counts[sim_idx] = similarity_counts.get(sim_idx, 0) + 1

    # Apply tiered bonuses for overlapping similarity
    for sim_idx, count in similarity_counts.items():
        bonus = base_bonus * (1 + (count - 1) * overlap_bonus_pct)
        sims[sim_idx] += bonus

    # Boost neighbors of similar games (proximity bonus)
    if similarity_counts:
        boosted_embeddings = [all_embeddings_np[idx] for idx in similarity_counts.keys()]
        boosted_center = np.mean(boosted_embeddings, axis=0)
        proximity_scores = cosine_sim(boosted_center, all_embeddings_np)
        sims += proximity_bonus * proximity_scores

    # Exclude input games themselves
    for idx in input_indices:
        sims[idx] = -np.inf

    # --- Apply Filters ---
    if filters is not None:
        valid_mask = np.ones(len(df), dtype=bool)

        # Exclude games released before year
        if filters.excludeBeforeYear is not None:
            try:
                year_threshold = int(filters.excludeBeforeYear)
                valid_mask &= df["release_date"].astype(str).str[:4].astype(int) >= year_threshold
            except Exception as e:
                print("Error filtering by year:", e)

        # Include Genres
        if filters.includeGenres:
            req_genres = set(t.lower().strip() for t in filters.includeGenres)
            valid_mask &= df["genres"].apply(
                lambda g: bool(req_genres.intersection(t.lower().strip() for t in g)) if isinstance(g, (list, tuple)) and g else False
            )

        # Exclude Genres
        if filters.excludeGenres:
            excl_genres = set(t.lower().strip() for t in filters.excludeGenres)
            valid_mask &= df["genres"].apply(
                lambda g: not bool(excl_genres.intersection(t.lower().strip() for t in g)) if isinstance(g, (list, tuple)) and g else True
            )

        # Include Themes
        if filters.includeThemes:
            req_themes = set(t.lower().strip() for t in filters.includeThemes)
            valid_mask &= df["themes"].apply(
                lambda t: bool(req_themes.intersection(x.lower().strip() for x in t)) if isinstance(t, (list, tuple)) and t else False
            )

        # Exclude Themes
        if filters.excludeThemes:
            excl_themes = set(t.lower().strip() for t in filters.excludeThemes)
            valid_mask &= df["themes"].apply(
                lambda t: not bool(excl_themes.intersection(x.lower().strip() for x in t)) if isinstance(t, (list, tuple)) and t else True
            )

        # Include Keywords
        if filters.includeKeywords:
            req_keys = set(t.lower().strip() for t in filters.includeKeywords)
            valid_mask &= df["keywords"].apply(
                lambda k: bool(req_keys.intersection(x.lower().strip() for x in k)) if isinstance(k, (list, tuple)) and k else False
            )

        # Exclude Keywords
        if filters.excludeKeywords:
            excl_keys = set(t.lower().strip() for t in filters.excludeKeywords)
            valid_mask &= df["keywords"].apply(
                lambda k: not bool(excl_keys.intersection(x.lower().strip() for x in k)) if isinstance(k, (list, tuple)) and k else True
            )

        # Platforms
        if filters.platforms:
            req_platforms = set(t.lower().strip() for t in filters.platforms)
            valid_mask &= df["platforms"].apply(
                lambda p: bool(req_platforms.intersection(x.lower().strip() for x in p)) if isinstance(p, (list, tuple)) and p else False
            )

        invalid_indices = np.where(~valid_mask)[0]
        sims[invalid_indices] = -np.inf
    # --- End Filter Section ---

    top_indices = np.argsort(sims)[::-1][:top_k]
    recommended_game_ids = [index_to_id[idx] for idx in top_indices]
    recommended_scores = sims[top_indices]

    # Filter out invalid -inf similarity scores before returning
    valid_recommendations = [
        (gid, float(score)) for gid, score in zip(recommended_game_ids, recommended_scores) if np.isfinite(score)
    ]

    return valid_recommendations

# def recommend_games(input_game_id, top_k=10, bonus_weight=0.2):
#     input_game_id = str(input_game_id) 

#     if input_game_id not in id_to_index:
#         print(f"Game ID {input_game_id} not found in dataset.")
#         return []

#     input_idx = id_to_index[input_game_id]
#     input_embedding = all_embeddings_np[input_idx]

#     sims = cosine_sim(input_embedding, all_embeddings_np)

#     # Add bonus similarity score for similar games
#     sim_indices = game_idx_to_similars.get(input_idx, [])
#     sims[sim_indices] += bonus_weight

#     sims[input_idx] = -np.inf  # exclude self

#     top_indices = np.argsort(sims)[::-1][:top_k]

#     recommended_game_ids = [index_to_id[idx] for idx in top_indices]
#     recommended_scores = sims[top_indices]
#     return list(zip(recommended_game_ids, recommended_scores))
