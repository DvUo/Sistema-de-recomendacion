import pandas as pd
from faker import Faker
import random
import chromadb
import json
from sentence_transformers import SentenceTransformer
from deepseek_api import get_summary

# Configuración inicial
fake = Faker()
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_movies():
    # Columnas reales del dataset MovieLens 100K (u.item)
    movies_cols = [
        "movie_id",       
        "title",           
        "release_date",    
        "video_release",   
        "imdb_url",        
        "unknown",         
        "Action",          
        "Adventure",       
        "Animation",       
        "Children",        
        "Comedy",          
        "Crime",           
        "Documentary",     
        "Drama",           
        "Fantasy",         
        "FilmNoir",        
        "Horror",          
        "Musical",         
        "Mystery",         
        "Romance",         
        "SciFi",           
        "Thriller",        
        "War",             
        "Western"          
    ]
    
    # Cargar el dataset con las columnas correctas
    movies = pd.read_csv(
        "./data/movies.tsv",
        sep="|",
        encoding="latin-1",
        header=None,
        names=movies_cols,
        engine="python"
    ).head(200)  
    
    # Limpieza de columnas no necesarias
    movies = movies.drop(columns=["release_date", "video_release", "imdb_url"])
    

    print("Generando resúmenes con DeepSeek...")
    movies["summary"] = movies["title"].apply(get_summary)
    print(movies["summary"].head())
    # Procesamiento de géneros (versión mejorada)
    genre_columns = movies_cols[6:]  # Todos los géneros desde la columna 5
    
    # Crear columna de géneros activos
    movies["genres"] = movies.apply(
        lambda x: ", ".join([genre for genre in genre_columns if x[genre] == 1]),
        axis=1
    )
    
    # Generar embeddings basados en géneros + resumen
    print("Generando embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combinamos géneros y resumen para mejor contexto
    movies["text_for_embedding"] = movies.apply(
        lambda x: f"Géneros: {x['genres']}. Resumen: {x['summary']}",
        axis=1
    )
    
    embeddings = model.encode(movies["text_for_embedding"].tolist())
    movies["embeddings"] = [embedding.tolist() for embedding in embeddings]
    
    return movies

def generate_users(movies):
    # Generar 30 usuarios con ratings aleatorios
    users = []
    movie_ids = movies["movie_id"].tolist()
    
    for user_id in range(1,31):
        ratings = {
            movie_id: random.randint(1, 5)
            for movie_id in random.sample(movie_ids, random.randint(20, 50))
        }
        
        # Crear vector de ratings completo
        rating_vector = [ratings.get(mid, 0) for mid in movie_ids]
        
        users.append({
            "user_id": user_id,
            "username": fake.user_name(),
            "ratings": ratings,
            "embeddings": rating_vector
        })
    
    return users

def save_to_chroma(movies, users):
    # Configurar ChromaDB persistente
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Colección de películas
    movies_collection = client.get_or_create_collection("movies")
    movies_collection.add(
        ids=[str(mid) for mid in movies["movie_id"]],
        embeddings=movies["embeddings"].tolist(),
        metadatas=movies[["title", "genres"]].to_dict("records")
    )
    
    # Colección de usuarios
    users_collection = client.get_or_create_collection("users")
    for user in users:
        users_collection.add(
            ids=str(user["user_id"]),
            embeddings=[user["embeddings"]],
            metadatas={
                "username": user["username"],
                "ratings": json.dumps(user["ratings"])
            }
        )

if __name__ == "__main__":
    movies = initialize_movies()
    users = generate_users(movies)
    save_to_chroma(movies, users)
    movies.to_csv("movies.csv", index=False)