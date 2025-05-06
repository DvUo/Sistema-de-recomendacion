from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from collections import defaultdict
from pydantic import BaseModel
import chromadb
import pandas as pd
import json
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
movies_df = pd.read_csv("movies.csv")

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    explanation: str

def get_recommendations(user_id: int):
    client = chromadb.PersistentClient(path="chroma_db")
    
    try:
        users_col = client.get_collection("users")
        
        # Obtener usuario objetivo
        user_data = users_col.get(ids=[str(user_id)])
        target_metadata = user_data['metadatas'][0]
        target_username = target_metadata['username']
        target_ratings = json.loads(target_metadata['ratings'])  # Convertir a dict

        # Obtener todos los usuarios (IDs 30)
        user_ids = [str(i) for i in range(1, 30)]
        all_users = users_col.get(ids=user_ids)
        
        # Comparar con otros usuarios
        similar_users = get_users_similar_to_target(target_ratings, all_users, target_username)
        
        # Convertir datos a matriz
        user_movie_matrix = convert_data_to_matrix(similar_users, target_ratings, target_username)

        # Calcular similitud
        similar_users_ponderation = get_ponderation_users(user_movie_matrix)

        if similar_users:
            # Obtener recomendaciones
            movie_recommendations = get_movie_recommendations(similar_users, target_ratings, similar_users_ponderation)
            get_message_recomendation(movie_recommendations)
            return movie_recommendations
        else:
            logger.info("No se encontraron usuarios similares.")
            return None
        
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return None

def convert_data_to_matrix(similar_users,target_ratings,target_username):
    # Convertir los datos a un DataFrame
    rows = []

    # Agregar el usuario objetivo
    for movie_id, rating in target_ratings.items():
        rows.append({
            "Username": target_username,
            "MovieId": movie_id,
            "Rating": rating
        })

    # Agregar usuarios similares
    for user in similar_users:
        username = user['username']
        ratings_dict = json.loads(user['ratings'] )  # Convertir a diccionario

        for movie_id, rating in ratings_dict.items():
            rows.append({
                "Username": username,
                "MovieId": movie_id,
                "Rating": rating
            })

    df = pd.DataFrame(rows)

    # Crear matriz de utilidad
    user_movie_matrix = df.pivot(index='Username', columns='MovieId', values='Rating').fillna(0)

    return user_movie_matrix

def get_users_similar_to_target(target_ratings, all_users,target_username):
    similar_users = []
    for user in all_users['metadatas']:
        if user['username'] == target_username:
            continue
        
        # Convertir ratings del usuario actual
        current_ratings = json.loads(user['ratings'])
        
        # Contar películas en común
        common_movies = set(target_ratings.keys()) & set(current_ratings.keys())
        if len(common_movies) > 5:
            similar_users.append(user)
    
    return similar_users

def get_ponderation_users(matrix_similar_users):
    
    similarities = cosine_similarity(matrix_similar_users)
    sim_df = pd.DataFrame(similarities, columns=matrix_similar_users.index, index=matrix_similar_users.index)
    return sim_df

def get_movie_recommendations(similar_users, target_ratings, matrix_similar_users):
    movies_watched = set(target_ratings.keys())
    score_sums = defaultdict(float)
    sim_sums = defaultdict(float)

    for user in similar_users:
        username = user['username']
        user_ratings = user['ratings']
        user_ratings = json.loads(user_ratings)
        similarity = matrix_similar_users.loc[username]  # fila completa
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in movies_watched:
                
                sim = similarity.get(username, 0)  # similitud con el target
                
                # Acumular puntuación ponderada y similitud
                score_sums[movie_id] += sim * rating
                sim_sums[movie_id] += sim

    recommendations = []
    for movie_id in score_sums:
        if sim_sums[movie_id] > 0:
            score = score_sums[movie_id] / sim_sums[movie_id]
            recommendations.append({
                'movieId': movie_id,
                'Score': int(score)
            })

    recommendations.sort(key=lambda x: x['Score'], reverse=True)
    return recommendations[:5]

def get_message_recomendation(recomendations):
    try:
        for data in recomendations:
            movie_id = data['movieId']
            
            score = data['Score']
            if int(score) > 3:
                movie = movies_df.loc[movies_df['movie_id'] == int(movie_id)]
                
                if not movie.empty:
                    return print("Segun la puntuacion de otros usuarios, la pelicula recomendada es: ", movie['title'].values[0]
                            , "de generos: " , movie['genres'].values[0], "con una puntuacion de: ", score, "y una sinopsis de: ", movie['summary'].values[0]) 
                else:
                    logger.warning(f"No se encontró la película con ID: {movie_id}")
            
    except Exception as e:
        logger.error(f"Error al obtener detalles de la película: {str(e)}")
        return None
get_recommendations(1)
# @app.post("/recommend", response_model=RecommendationResponse)
# async def recommend(request: RecommendationRequest):
#     recommendations = get_recommendations(request.user_id, request.top_n)
    
#     if recommendations is None or len(recommendations) == 0:
#         return RecommendationResponse(
#             user_id=request.user_id,
#             recommendations=[],
#             explanation="No se encontró el usuario o no hay recomendaciones disponibles."
#         )
    
#     explanation = (
#         "Recomendaciones basadas en las valoraciones de usuarios con patrones de visualización similares. "
#         "Las películas son seleccionadas por su alta calificación promedio entre usuarios similares."
#     )
    
#     return RecommendationResponse(
#         user_id=request.user_id,
#         recommendations=recommendations,
#         explanation=explanation
#     )

# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)