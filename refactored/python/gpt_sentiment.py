import pandas as pd
import re
import numpy as np

def gpt_sentiment(df, 
                  system_message="You are an expert sentiment analyzer, analyzing emotion from text. Please carefully read the following text and respond with '-1' if the sentiment is negative, '0' if it's neutral, or '1' if it's positive.",
                  temperature=0.2):
    sentiment_list = []
    flags = []
    
    for i in range(len(df)):
        # Check if the text input is not empty
        if len(df['Text'][i]) > 0:
            sentiment = gpt_api(df['Text'][i], 
                                system_message = system_message, 
                                model = "gpt-3.5-turbo", 
                                temperature = temperature, 
                                max_tokens = 2000, 
                                num_retries = 3, 
                                pause_base = 1, 
                                presence_penalty = 0.0, 
                                frequency_penalty = 0.0)
            
            # Filter the sentiment value
            sentiment_numbers = list(map(int, re.findall("-1|0|1", sentiment)))
            
            if len(sentiment_numbers) == 0:
                flags.append("Invalid number")
                sentiment_list.append(np.nan)
            else:
                sentiment_list.append(sentiment_numbers)
                flags.append("Multiple valid numbers" if len(sentiment_numbers) > 1 else "Single valid number")
        else:
            flags.append("Empty text")
            sentiment_list.append(np.nan)
    
    result = pd.DataFrame({'ID': df['ID'], 'sentiment': sentiment_list, 'flag': flags})
    return result
