# Sistema de Recomendación de Películas

Este proyecto implementa un sistema de recomendación de películas basado en la similitud de usuarios y películas, utilizando embeddings y una base de datos vectorial persistente. El sistema está diseñado para ser ejecutado en un entorno local y expone una API REST para obtener recomendaciones personalizadas.

## Estructura del Proyecto

- `main.py`: Archivo principal que expone la API de recomendaciones utilizando FastAPI. Contiene la lógica para calcular recomendaciones basadas en la similitud de usuarios.
- `generate_data.py`: Script encargado de generar y procesar los datos de películas y usuarios. Se debe ejecutar primero para crear o actualizar la base de datos de películas y usuarios.
- `deepseek_api.py`: Módulo que interactúa con la API de DeepSeek para generar resúmenes automáticos de las películas.
- `movies.csv`: Archivo generado automáticamente que contiene la información procesada de las películas, incluyendo géneros, resúmenes y embeddings.
- `chroma_db/`: Carpeta que almacena la base de datos vectorial persistente utilizada por ChromaDB.
- `data/movies.tsv`: Archivo fuente con los datos originales de las películas (formato MovieLens).
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `.env`: Archivo de variables de entorno, donde se debe colocar la clave de API de DeepSeek.
> En caso de querer actualizar la base de datos recordar crear el archivo .env e incluir su api key -> DEEPSEEK_API_KEY= api-key

---

## Descripción de Archivos

### 1. `generate_data.py`

Este script debe ejecutarse en primer lugar. Su función es:

- Leer el archivo de películas original (`data/movies.tsv`).
- Generar resúmenes automáticos de cada película utilizando la API de DeepSeek.
- Procesar los géneros y crear embeddings combinando géneros y resúmenes.
- Generar usuarios ficticios con ratings aleatorios para simular interacciones.
- Guardar toda la información procesada en `movies.csv`.
- Almacenar los embeddings y metadatos en la base de datos vectorial persistente (`chroma_db/`).

**Ejemplo de uso:**

```bash
python generate_data.py
```
  > Se recomienda ejecutar este script cada vez que se agreguen nuevas películas o se desee actualizar los datos y embeddings.

### 2. `deepseek_api.py` 
Este módulo encapsula la lógica para interactuar con la API de DeepSeek. Permite generar resúmenes automáticos de películas a partir de su título. Si ocurre un error durante la generación del resumen, retorna un mensaje por defecto.
-  No requiere ejecución directa, ya que es utilizado internamente por `generate_data.py`.

### 3. `main.py`
Este archivo implementa el servidor de la API utilizando FastAPI. Sus responsabilidades incluyen:

Cargar los datos de películas desde movies.csv.

- Conectarse a la base de datos vectorial (chroma_db/) para obtener información de usuarios.

- Calcular la similitud entre usuarios utilizando la métrica de coseno.

- Generar recomendaciones personalizadas para un usuario dado, basándose en los ratings de usuarios similares.

- Exponer un endpoint /recommend que recibe un user_id y retorna una lista de películas recomendadas, junto con sus géneros, resumen y puntuación estimada.

- Incluir un endpoint /health para verificar el estado del servidor.

```bash
    python main.py
```
  > Luego, se puede realizar una petición POST al endpoint /recommend.

### Flujo de uso
1. Preparar el entorno:

  - Instalar las dependencias necesarias utilizando:

```bash
    pip install -r requirements.txt
```
  - Asegurarse de tener la clave de API de DeepSeek en el archivo .env.

2. Generar y actualizar los datos:

  - Ejecutar generate_data.py para procesar las películas, generar resúmenes y embeddings, y poblar la base de datos vectorial.

3. Iniciar el servidor de la API:

  - Ejecutar main.py con Uvicorn para exponer los endpoints de recomendación.

4. Consumir la API:

  - Realizar peticiones al endpoint /recommend para obtener recomendaciones personalizadas.
---
### Ejemplo de Petición y Respuesta
  #### Petición
  ``` http
    POST /recommend
    Content-Type: application/json
    
    {
      "user_id": 10
    }
```
  #### Respuesta
  ``` json
    {
    "message": "Te recomendamos las siguientes películas según la calificación de usuarios similares:",
    "recommendations": [
      {
        "title": "Toy Story (1995)",
        "genres": "Animation, Children, Comedy",
        "score": 5,
        "summary": "Toy Story sigue a Woody, el juguete favorito de Andy, cuyo liderazgo se ve amenazado con la llegada del nuevo y moderno Buzz Lightyear, llevándolos a una aventura llena de rivalidad, amistad y descubrimientos."
      }
    ]
  }
```
---
### Notas adicionales

 - Si se agregan nuevas películas o se desea actualizar los resúmenes/embeddings, se debe volver a ejecutar generate_data.py.
  
 - El sistema utiliza usuarios ficticios generados aleatoriamente para simular el comportamiento de un sistema real.
  
 - La clave de API de DeepSeek es obligatoria para la generación de resúmenes automáticos.


