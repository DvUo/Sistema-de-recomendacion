# main.py (FastAPI)
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import pandas as pd
import json

app = FastAPI()
movies_df = pd.read_csv("movies.csv")

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    explanation: str

def get_recommendations(user_id: int, top_n: int):
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Obtener usuario objetivo
    users_collection = client.get_collection("users")
    user_data = users_collection.get(ids=[str(user_id)])
    
    if not user_data["ids"]:
        return None
    
    # Buscar usuarios similares
    similar_users = users_collection.query(
        query_embeddings=user_data["embeddings"],
        n_results=11
    )
    
    # Procesar ratings
    target_ratings = json.loads(user_data["metadatas"][0]["ratings"])
    movie_ids = movies_df["movieId"].tolist()
    
    # Calcular puntuaciones
    movie_scores = {}
    for i, similar_id in enumerate(similar_users["ids"][0]):
        if similar_id == str(user_id):
            continue
        
        ratings = json.loads(similar_users["metadatas"][0][i]["ratings"])
        for mid, score in ratings.items():
            if mid not in target_ratings:
                movie_scores.setdefault(mid, []).append(score)
    
    # Generar recomendaciones
    recommendations = []
    for mid, scores in movie_scores.items():
        avg_score = sum(scores) / len(scores)
        movie_title = movies_df[movies_df["movieId"] == int(mid)]["title"].values[0]
        recommendations.append({
            "movie_id": mid,
            "title": movie_title,
            "average_score": round(avg_score, 2),
            "ratings_count": len(scores)
        })
    
    return sorted(recommendations, key=lambda x: (-x["average_score"], -x["ratings_count"]))[:top_n]

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    recommendations = get_recommendations(request.user_id, request.top_n)
    
    explanation = (
        "Recomendaciones basadas en las valoraciones de usuarios con patrones de visualización similares. "
        "Las películas son seleccionadas por su alta calificación promedio entre usuarios similares."
    )
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        explanation=explanation
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)