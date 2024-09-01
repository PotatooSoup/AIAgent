from openai import OpenAI
import pandas as pd
import requests

client = OpenAI()

input_csv_file = 'MSc-data/test.csv'
output_csv_file = 'MSc-data/output_summaries.csv'

df = pd.read_csv(input_csv_file)

def generate_summary(dialogue, model_name):
    if model_name.startswith('gpt'):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a natural paragraph summary based on the conversation between a doctor and a patient. Including the patient's age, presenting symptoms, duration of symptoms, and any specific details provided. Do not make up information."
                },
                {
                    "role": "user",
                    "content": dialogue
                }
            ],
            temperature=0.7,
            max_tokens=150,
            top_p=1
        )
        return response.choices[0].message.content
    elif model_name.startswith('Qwen'):
        url = "<LLM_url>"
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Generate a natural paragraph summary based on the conversation between a doctor and a patient. Including the patient's age, presenting symptoms, duration of symptoms, and any specific details provided. Do not make up information."
                },
                {
                    "role": "user",
                    "content": dialogue
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "<API Key>"
        }
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        raise ValueError("Unsupported model name")

df['Model_A'] = df['dialogue'].apply(lambda x: generate_summary(x, "gpt-4o-mini"))

df['Model_B'] = df['dialogue'].apply(lambda x: generate_summary(x, "Qwen/Qwen2-72B-Instruct"))

df.to_csv(output_csv_file, index=False)
