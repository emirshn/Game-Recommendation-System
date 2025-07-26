from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import databases
import sqlalchemy
import model_utils
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'games.db')}"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from pydantic import BaseModel
from typing import List, Optional

class RecommendationOptions(BaseModel):
    excludeBeforeYear: Optional[int] = None
    limit: Optional[int] = 50
    excludeKeywords: List[str] = []
    includeKeywords: List[str] = []
    includeGenres: List[str] = []
    excludeGenres: List[str] = []
    includeThemes: List[str] = []
    excludeThemes: List[str] = []
    platforms: List[str] = []

class RecommendationRequest(BaseModel):
    selected_games: List[int]
    options: RecommendationOptions


class SearchRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup():
    await database.connect()
    model_utils.load_model_and_data()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/api/recommend")
async def recommend(request: RecommendationRequest):
    try:
        selected_games = request.selected_games
        print(request)
        if not selected_games:
            return {"recommendations": []}

        input_game_ids = selected_games
        options = request.options
        top_k = options.limit

        recommended_ids_scores = model_utils.recommend_games(input_game_ids, top_k=top_k, filters=options)
        recommended_ids = [gid for gid, score in recommended_ids_scores]

        if not recommended_ids:
            return {"recommendations": []}

        placeholders = ", ".join([f":id_{i}" for i in range(len(recommended_ids))])
        params = {f"id_{i}": gid for i, gid in enumerate(recommended_ids)}

        query = f"""
            SELECT
              g.*,
              GROUP_CONCAT(DISTINCT gg.genre) AS genres,
              GROUP_CONCAT(DISTINCT gt.theme) AS themes,
              GROUP_CONCAT(DISTINCT gf.franchise) AS franchise,
              GROUP_CONCAT(DISTINCT gs.series) AS series,
              GROUP_CONCAT(DISTINCT gmd.main_developer) AS main_developers,
              GROUP_CONCAT(DISTINCT gsd.supporting_developer) AS supporting_developers,
              GROUP_CONCAT(DISTINCT gp.publisher) AS publishers,
              GROUP_CONCAT(DISTINCT gpl.platform) AS platforms,
              GROUP_CONCAT(DISTINCT gpp.player_perspective) AS player_perspectives,
              GROUP_CONCAT(DISTINCT gmode.game_mode) AS game_modes,
              GROUP_CONCAT(DISTINCT gengine.game_engine) AS game_engines,
              GROUP_CONCAT(DISTINCT gsim.similar_game) AS similar_games,
              GROUP_CONCAT(DISTINCT gkey.keyword) AS keywords,
              GROUP_CONCAT(DISTINCT gss.screenshot_url) AS screenshot_urls,
              GROUP_CONCAT(DISTINCT gar.artwork_url) AS artwork_urls
            FROM games g
            LEFT JOIN game_genres gg ON g.id = gg.game_id
            LEFT JOIN game_themes gt ON g.id = gt.game_id
            LEFT JOIN game_franchises gf ON g.id = gf.game_id
            LEFT JOIN game_series gs ON g.id = gs.game_id
            LEFT JOIN game_main_developers gmd ON g.id = gmd.game_id
            LEFT JOIN game_supporting_developers gsd ON g.id = gsd.game_id
            LEFT JOIN game_publishers gp ON g.id = gp.game_id
            LEFT JOIN game_platforms gpl ON g.id = gpl.game_id
            LEFT JOIN game_perspectives gpp ON g.id = gpp.game_id
            LEFT JOIN game_modes gmode ON g.id = gmode.game_id
            LEFT JOIN game_engines gengine ON g.id = gengine.game_id
            LEFT JOIN game_similar_games gsim ON g.id = gsim.game_id
            LEFT JOIN game_keywords gkey ON g.id = gkey.game_id
            LEFT JOIN game_screenshots gss ON g.id = gss.game_id
            LEFT JOIN game_artworks gar ON g.id = gar.game_id
            WHERE g.id IN ({placeholders})
            GROUP BY g.id
            LIMIT :limit
        """

        params["limit"] = top_k

        rows = await database.fetch_all(query, values=params)

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "release_date": row["release_date"],
                "genres": row["genres"] or "",
                "themes": row["themes"] or "",
                "franchise": row["franchise"] or "",
                "series": row["series"] or "",
                "user_score": row["rating"] or "",
                "critic_score": row["aggregated_rating"] or "",
                "main_developers": row["main_developers"] or "",
                "supporting_developers": row["supporting_developers"] or "",
                "publishers": row["publishers"] or "",
                "platforms": row["platforms"] or "",
                "player_perspectives": row["player_perspectives"] or "",
                "game_modes": row["game_modes"] or "",
                "game_engines": row["game_engines"] or "",
                "similar_games": row["similar_games"] or "",
                "keywords": row["keywords"] or "",
                "cover_url": row["cover_url"] or "",
                "summary": row["summary"] or "",
                "screenshot_urls": (row["screenshot_urls"] or "").split(",") if row["screenshot_urls"] else [],
                "artwork_urls": (row["artwork_urls"] or "").split(",") if row["artwork_urls"] else [],
            })

                # After query and before return
        
        id_to_result = {str(game["id"]): game for game in results}
        print("Recommended IDs:", recommended_ids_scores)
        sorted_results = []
        for gid, score in recommended_ids_scores:
            game = id_to_result.get(str(gid))  
            if game:
                game["rec_score"] = float(score)
                sorted_results.append(game)

        return {"recommendations": sorted_results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

from fastapi import Query

@app.get("/api/search")
async def search(query: str = Query(..., min_length=1)):
    like_query = f"%{query.lower()}%"
    query_sql = """
        SELECT id, name, release_date, cover_url
        FROM games
        WHERE LOWER(name) LIKE :query
        ORDER BY rating DESC NULLS LAST
        LIMIT 20;
    """
    rows = await database.fetch_all(query=query_sql, values={"query": like_query})
    return [{"id": row["id"], "name": row["name"], "release_date": row["release_date"], "cover_url": row["cover_url"]} for row in rows]

@app.get("/api/options")
async def get_filter_options():
    try:
        queries = {
            "keywords": "SELECT DISTINCT keyword FROM game_keywords ORDER BY keyword;",
            "genres": "SELECT DISTINCT genre FROM game_genres ORDER BY genre;",
            "platforms": "SELECT DISTINCT platform FROM game_platforms ORDER BY platform;",
            "themes": "SELECT DISTINCT theme FROM game_themes ORDER BY theme;"
        }

        results = {}

        for key, sql in queries.items():
            rows = await database.fetch_all(sql)
            results[key] = [row[0] for row in rows if row[0]]  

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
