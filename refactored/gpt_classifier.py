import pandas as pd
import os
import requests
import json
import re
import time
from typing import List, Tuple

# Set API key as environment variable
def set_api_key(api_key):
    if api_key is None:
        raise ValueError("API key is not set.")
    os.environ['OPENAI_API_KEY'] = api_key

set_api_key("sk-YOUR-API-KEY")

# Reshape a DataFrame into a new DataFrame format
def reshape_df(df, n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Error: n must be a positive integer.")

    new_df = pd.DataFrame(columns=["variables", "responses"])

    for col in df.columns:
        responses = df[col].dropna()[:min(n, len(df[col].dropna()))]
        responses_str = ", ".join([str(r) for r in responses])
        new_df = new_df.append({"variables": col, "responses": responses_str}, ignore_index=True)

    return new_df

# Make API call to OpenAI GPT
def gpt_api(prompt, model="gpt-3.5-turbo", temperature=0.5, max_tokens=50, system_message=None, num_retries=3):
    if os.getenv('OPENAI_API_KEY') is None:
        raise ValueError("API key is not set.")
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "user", "content": prompt}]

    if system_message is not None:
        messages.insert(0, {"role": "system", "content": system_message})

    data = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages
    }

    for _ in range(num_retries):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            break
        time.sleep(1)  # Add delay between retries

    if response.status_code != 200:
        raise Exception(f"Failed to call GPT API. Error message: {response.json().get('error', 'Unknown error')}")

    response_content = response.json()

    if len(response_content["choices"]) > 0:
        message = response_content["choices"][0]["message"]["content"]
    else:
        message = "The model did not return a message. You may need to increase max_tokens."

    clean_message = message.replace("\n", " ").strip()

    return clean_message

def gpt_classifier(df: pd.DataFrame, n: int, system_message: str = "You are an expert data type classifier AI. Please carefully read the following series of values and respond with only a single numeric value representing the type of data it is. Respond with '1' if it's numeric (unique values all numbers), '2' if it's categorical (an alternating value or dog, the column tends to have low uniqueness among responses, any variable primarily going to be used as a category), '3' if it's character (names, words, strings of letters or cells with only tex), '4' if it's a date (regardless of format 'January 1st', 'Nov-1999', '11-01-1999'), '5' if it's logical (only TRUE/FALSE or T/F), or '6' if it's geospatial (coordinates). If you are unsure provide your best guess as to how the data would be used analytically which affords the most freedom.", temperature = 0.2) -> pd.DataFrame:
    
    df = reshape_df(df, n)
    
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise ValueError("The data frame must have at least one row and two columns")
    
    dtype_list = []
    flags = []
    gpt_responses = []
    
    for i in range(df.shape[0]):
        dtype = gpt_api(df["responses"].iloc[i],
                        system_message=system_message, 
                        model="gpt-3.5-turbo", 
                        temperature=temperature, 
                        max_tokens=2000, 
                        num_retries=3)
        
        gpt_responses.append(dtype)
        
        dtype_numbers = [int(x) for x in re.findall(r'1|2|3|4|5|6', dtype)]
        
        if len(dtype_numbers) == 0:
            flags.append("Invalid number")
            dtype_list.append(None)
        else:
            dtype_list.append(dtype_numbers[0])
            flags.append("Multiple valid numbers" if len(dtype_numbers) > 1 else "Single valid number")
    
    dtype_labels = ['numeric', 'categorical', 'character', 'date', 'logical', 'geospatial']
    data_types = [dtype_labels[i-1] if i is not None else None for i in dtype_list]
    
    result = pd.DataFrame({
        'Variables': df['variables'], 
        'Responses': df['responses'], 
        'GptResponses': gpt_responses, 
        'DataType': data_types, 
        'Flag': flags
    })
    
    return result
