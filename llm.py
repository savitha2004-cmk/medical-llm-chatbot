import requests

def ask_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",   # change to "phi" if slow
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()
        answer = data.get("response")

        if not answer or answer.strip() == "":
            return "NOT FOUND"

        return answer

    except Exception as e:
        return f"LLM Error: {str(e)}"