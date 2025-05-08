from openai import OpenAI
import dotenv
api_key = dotenv.dotenv_values(".env")["DEEPSEEK_API_KEY"]

def summarize_with_deepseek(text, prompt_extra="Resume en no más de 3 líneas la película según el título que te adjuntare a continuacion: "):
    # Utiliza la API de DeepSeek para generar un resumen breve de una película a partir de su título.
    
    client = OpenAI(api_key, base_url="https://api.deepseek.com")
    prompt = f"{prompt_extra}\n\n{text}"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en películas y en resumir texto"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def get_summary(title):
    # Llama a summarize_with_deepseek y maneja errores, devolviendo un resumen para el título dado.
    
    try:
        return summarize_with_deepseek(title)
    except Exception as e:
        print(f"Error al procesar {title}: {e}")
        return "Resumen no disponible"
