import os
import json
import requests
import pandas as pd
from dateutil.parser import parse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Set API key
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

set_api_key("sk-your-api-key")

def gpt_api(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50, 
            system_message = None, num_retries = 3, pause_base = 1, 
            presence_penalty = 0.0, frequency_penalty = 0.0):
    
    messages = [{"role": "user", "content": prompt}]
    if system_message is not None:
        messages.insert(0, {"role": "system", "content": system_message})

    session = requests.Session()
    retry = Retry(total=num_retries, backoff_factor=pause_base)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    response = session.post(
        url = "https://api.openai.com/v1/chat/completions",
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        },
    )
    
    response.raise_for_status()

    content = response.json()
    if len(content['choices']) > 0:
        message = content['choices'][0]['message']['content']
    else:
        message = "The model did not return a message. You may need to increase max_tokens."

    clean_message = message.replace("\n", " ").strip()

    return clean_message

def gpt_date_parser(df, column, system_message = "You are an expert date identifier. You will be given text which you must read and determine if it's a real date. If it is a real date, restate it in YYYY-MM-DD format, surrounded with '#'. If there is no date, state that there is no date.", temperature = 0.1):
    
    parsed_df = pd.DataFrame(columns=["id", "original", "parsed", "status", "parsed_by"])
    failed_df = pd.DataFrame(columns=["id", "original", "parsed", "status", "parsed_by"])
    
    for i, row in df.iterrows():
        id = row['id']
        date = row[column]

        try:
            parsed_date = parse(date)
            parsed_df = parsed_df.append({
                "id": id, 
                "original": date, 
                "parsed": parsed_date.strftime('%Y-%m-%d'), 
                "status": "parsed", 
                "parsed_by": "Python"}, 
                ignore_index=True)
            print(f"Parsed: {date}")
        except:
            failed_df = failed_df.append({
                "id": id, 
                "original": date, 
                "parsed": None, 
                "status": "failed", 
                "parsed_by": "NA"}, 
                ignore_index=True)
            print(f"Failed to parse: {date}")
    
    for i, row in failed_df.iterrows():
        original_date = row['original']
        prompt = f"What date is specified in '{original_date}'?"

        response = gpt_api(prompt, system_message=system_message, temperature=temperature)
        response_date = None
        if '#' in response:
            response_date = response.split('#')[1].strip()

        if response_date is not None:
            try:
                final_date = parse(response_date)
                failed_df.at[i, 'parsed_by'] = 'GPT'
                failed_df.at[i, 'parsed'] = final_date.strftime('%Y-%m-%d')
            except:
                failed_df.at[i, 'parsed_by'] = 'NA'
        else:
            failed_df.at[i, 'parsed_by'] = 'NA'

        print(f"Response from GPT: {response_date}")

    result_df = pd.concat([parsed_df, failed_df])
    result_df.sort_values(by=['id'], inplace=True)

    return result_df