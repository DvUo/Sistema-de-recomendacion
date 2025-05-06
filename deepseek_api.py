from openai import OpenAI

def summarize_with_deepseek(text, prompt_extra="Resume en no más de 3 líneas la película según el título que te adjuntare a continuacion: "):
    client = OpenAI(api_key="sk-6c32bff31fa04637b47b54d31b7b5f95", base_url="https://api.deepseek.com")
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

# Generar resúmenes con DeepSeek (como en tu código original)
def get_summary(title):
    try:
        return summarize_with_deepseek(title)
    except Exception as e:
        print(f"Error al procesar {title}: {e}")
        return "Resumen no disponible"
