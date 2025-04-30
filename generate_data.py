# generate_data.py
import pandas as pd
from faker import Faker
import random
import chromadb
import json
from sentence_transformers import SentenceTransformer

# Configuración inicial
fake = Faker()
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_movies():
    # Cargar y procesar datos de películas
    movies_cols = ["movieId", "title"] + [str(i) for i in range(19)]
    movies = pd.read_csv("movies.tsv", sep="|", names=movies_cols, encoding="latin-1", engine="python").head(200)
    
    # Generar resúmenes y embeddings
    movies["genres"] = movies.apply(lambda x: " ".join([str(col) for col in movies_cols[2:] if x[col] == 1]), axis=1)
    movies["embeddings"] = model.encode(movies["genres"].tolist()).tolist()
    
    return movies

def generate_users(movies):
    # Generar 30 usuarios con ratings aleatorios
    users = []
    movie_ids = movies["movieId"].tolist()
    
    for user_id in range(30):
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
        ids=[str(mid) for mid in movies["movieId"]],
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