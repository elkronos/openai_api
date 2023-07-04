import requests

def whisper_transcribe(file_path, api_key, model="whisper-1", response_format="json", temperature=0, language=None, prompt=None):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "model": model,
        "response_format": response_format,
        "temperature": temperature
    }
    if language is not None:
        params["language"] = language
    if prompt is not None:
        params["prompt"] = prompt
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, headers=headers, params=params, files=files)
    return response.json()

def whisper_translate(file_path, api_key, model="whisper-1", response_format="json", temperature=0, prompt=None):
    url = "https://api.openai.com/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "model": model,
        "response_format": response_format,
        "temperature": temperature
    }
    if prompt is not None:
        params["prompt"] = prompt
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, headers=headers, params=params, files=files)
    return response.json()
