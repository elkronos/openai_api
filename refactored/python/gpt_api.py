import os
import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# Set API key as system variable
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

set_api_key("sk-paste-your-api-key-here")
print(os.environ["OPENAI_API_KEY"])

def create_headers():
    return {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

def create_data(prompt, model, temperature, max_tokens, messages, presence_penalty, frequency_penalty):
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty
    }

def handle_response(response):
    if len(response['choices']) > 0:
        message = response['choices'][0]['message']['content']
    else:
        message = "The model did not return a message. You may need to increase max_tokens."
    return message.replace("\n", " ").strip()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def send_request(headers, data):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()

def gpt_api(prompt, model="gpt-3.5-turbo", temperature=0.5, max_tokens=50, 
            system_message=None, presence_penalty=0.0, frequency_penalty=0.0):

    messages = [{"role": "user", "content": prompt}]
    if system_message is not None:
        messages.insert(0, {"role": "system", "content": system_message})
    
    headers = create_headers()
    data = create_data(prompt, model, temperature, max_tokens, messages, presence_penalty, frequency_penalty)

    response_json = send_request(headers, data)
    
    return handle_response(response_json)
