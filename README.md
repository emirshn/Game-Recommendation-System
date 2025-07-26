# Game-Recommendation-System
This project is an end-to-end content based Game Recommendation System powered by a large IGDB-based game dataset (350k+ games) and an interactive Vue 3 frontend with a Python backend.
It provides personalized recommendations based on games the user selects, with filtering features.

---

## Features
- **Multi-game input** (up to 3)
- **Hybrid recommendation scoring**:
  - Tag embedding similarity
  - Autoencoder-based preference learning
  - Overlap and proximity bonuses
- **Advanced options**:
  - Exclude and include genres, themes and keywords
  - Minimum release date
  - Platform selection
- **Visual previews**:
  - Screenshot and artwork modals with arrows
  - Name, Release date, rating, critic rating, genres, themes, franchise, series, main and supporting developers, publishers, platforms, player perspectives, game modes, game engines, keywords, cover image, summmary, screenshot and artwork images can be seen if they exist in database for given game
- **Sorting options**:
  - User rating, critic score, release date, recommendation score (asc/desc)
---
## Project Structure
- ├── backend/
- │ ├── backend # API server (FastAPI)
- │ ├── model_generator # Used for training models and generating embedding files for future use without effecting actual app
- │ ├── model_utils # Same as generator but it is used by backend.py for calculating recommendations from existing embedded files
- │ └── database as csv and db files are also stored here with output of model generator (game_embeddings, tag_vocab etc.)
- ├── frontend/
- │ ├── public/ 
- │ ├── components/ # Components are used for clean code
- │ ├── App.vue 
- │ └── main.js 
---
## Setup
- For using this app first you need a game_dataset.csv and games.db (db version of csv file). Because of upload problems i didnt include these to project directly (you can still unzip game_dataset and gamesdb files in backend folder) but you can also generate yours using code in this [repo](https://github.com/emirshn/IGDB-Database-Fetcher) depending on your preferences (for example i didnt include games under 50 user score and ports etc.)
- Next if you created a new dataset you need: game_embeddings.npy, game_idx_to_similars.pkl, game_idx_to_similars.pth, tag_vocab.pkl files which can be generated from model_generator.py using:
  " python model_generator.py train " command. Or you can unzip model_files.zip and put them into backend folder if you used given dataset from [repo](https://github.com/emirshn/IGDB-Database-Fetcher).
- For running frontend:
  - cd frontend
  - npm install
  - npm run dev
- For running backend:
  - install libraries with pip
  - cd backend
  - uvicorn backend:app --reload
- Update any API URLs inside your frontend code to match your backend's running port (e.g., http://localhost:5000 or 127.0.0.1:8000)
---

## Usage Tips
- Pick 1 to 3 favorite games.
- Use Advanced Options to improve precision. 
- Apply filters to avoid noisy or overly broad recommendations.
- Click game cards to view large images and explanations.
  
---
## How Training and Recommendations Work ?
1. Data Processing
-The system loads a dataset (game_dataset.csv) containing structured tags like: genres, themes, keywords, game_modes, player_perspectives, and similar_games.

3. Vocabulary Building
- For each tag type (e.g., genres), a unique vocabulary is built. Tags are mapped to unique indices to create input features for the model.

3. TF-IDF Weighting
- Each tag is assigned an IDF (Inverse Document Frequency) score to weigh rarer tags more heavily. These IDF scores are used as input weights during training.

4. Autoencoder Model
- A PyTorch-based autoencoder is used to embed each game into a fixed-length (128-dim) vector.
- Inputs are weighted averages of tag embeddings per category.
- The model is trained to reconstruct these representations, learning a compressed semantic representation.

5. Embeddings Generation
- After training, the model creates embeddings for every game. These embeddings are stored in game_embeddings.npy.
- A separate mapping of similar_games is also saved for post-processing.

6. Game Recommendation
- Given one or more game IDs, the system:
  - Embeds input games using their vector representations.
  - Computes cosine similarity between input and all games.
- Applies tiered bonuses for games that:
  - Are listed as "similar games" in the input.
  - Are shared similar games between multiple inputs.
  - Are close in the embedding space.
- Applies user-defined filters:
  - Include/Exclude by genres, themes, keywords, platforms
  - Exclude games released before a given year.
  - Returns top K recommended games with similarity scores.

