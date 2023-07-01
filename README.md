# openai_api
Interact with openai APIs using R (DALLE2, GPT, Whisper)

# Scripts

Below is a summary of each script. You can find detailed explanations, optional parameter defaults, and examples in the roxygen located within each script.

## 01 - `dalle_api.R` - Contains several functions related to making API calls to OpenAI and generating and manipulating images using the DALL·E model. 

Here's a brief overview of each function in this script:

* `dalle_call`: This function is a common handler for making API calls to OpenAI. It takes parameters such as the API endpoint, HTTP method, headers, body, and query, and returns the content of the API response as a list.
* `dalle_generate`: This function handles API calls to generate and download images based on given prompts. It makes a POST request to the "images/generations" endpoint, generates images based on the provided prompt, and downloads the images to the specified path. It also updates a global dataframe with the prompt and timestamp.
* `save_prompts`: This function saves a dataframe containing prompts and timestamps to a specified CSV file.
* `dalle_variation`: This function generates variations of an image by applying transformations such as scaling, modulating brightness, and applying blur. The resulting image is saved to disk.

## 02 - `fine_tune_model.R` - Allows you to fine-tune a GPT (Generative Pre-trained Transformer) model using the OpenAI API. The function takes several parameters such as the training file ID, validation file ID, model name, number of epochs, batch size, learning rate multiplier, prompt loss weight, and other optional parameters.

## 03 - `gpt_api.R` - This script provides functions for interacting with OpenAI's API to generate text responses. 

Here's a brief overview of each function in this script:

* `gpt_api`: This function sends a prompt to the OpenAI API and returns the generated text. It handles API communication, error checking, and basic formatting of the response. The function takes several parameters such as the prompt, model to use, temperature, maximum tokens, system message, and penalties for new tokens. It makes a POST request to the OpenAI API endpoint and retrieves the generated text response.
* `count_tokens`: This function estimates the number of tokens in a given text string. It splits the text into words, punctuation, and whitespace and returns the count of tokens. Note that this is an approximation and may not match the token count used by specific language models.
* `estimate_cost`: This function calculates the estimated cost based on the number of input and output tokens and the specified model. It takes the number of input tokens, number of output tokens, and the model as input, and returns the estimated cost for the input tokens, output tokens, and the total cost.

## 04 - `gpt_personality.R` - The script defines two functions: gpt_personality and score_BFI.

Here's a brief overview of each function in this script:

*`gpt_personality`: The gpt_personality function is used to analyze personality traits based on a given text (lyrics). It takes a data frame as input, which should contain a column named "Text" with the lyrics to be analyzed. The function applies OpenAI's GPT model to evaluate how the author of the lyrics would respond to certain statements regarding their personality. Each statement is evaluated on a scale from 1 (disagree strongly) to 5 (agree strongly). The function uses the gpt_api function, which should be loaded separately, and requires an API key with associated cost.
*`score_BFI`: The score_BFI function calculates the Big Five Inventory (BFI) scores based on the personality analysis performed by the gpt_personality function. It reverse scores relevant items and calculates the mean scores for each personality trait (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness to Experience). The function also computes Cronbach's Alpha for each personality trait.

## 05 - gpt_read.R - This script is a collection of functions for processing and analyzing text using the GPT (Generative Pre-trained Transformer) model from OpenAI. 

Here's a brief description of each function:

*`parse_text`: This function takes different types of input files (PDF, DOCX, TXT) and performs optical character recognition (OCR) on scanned documents and images. It converts the text into tokens, removes whitespace, special characters, and numbers, and splits the text into chunks of tokens to avoid exceeding the model's token limit.
*`search_text`: This function extracts passages from a given text that contain specified keywords. It searches for each keyword in the text and returns a list of passages (windows) that surround the occurrences of the keywords.
*`gpt_read`: This function reads a list of text chunks and a question, and uses the GPT model to generate a response based on the text and question. It sends requests to the OpenAI API, processes the responses, and returns the generated response from the GPT model. It also includes additional features such as system messages, error handling, and parameter tuning.


## 06 - gpt_search_api.R - This script provides a set of functions for performing grid searches using the GPT API. 

Here is a brief description of each function:

* `call_api`: This function calls the OpenAI GPT API to generate a message using a specified language model. It handles retries, error checking, and processes the response from the API.
* `gpt`: This function uses a specified GPT model to generate text based on a given prompt. It can include system messages in the conversation and is built on top of the call_api function.
* `gpt_search`: This function performs a grid search over specified GPT models and parameters using a given prompt. It generates results for various combinations of models, temperatures, max tokens, and system messages. Results can be saved to disk in batches.
* `import_rds_files`: This function sets the working directory to a specified path and imports RDS files from that directory. It returns a list of data frames containing the imported data.

## 07 - gpt_sentiment.R - Performs sentiment analysis on a given dataframe using the GPT from the OpenAI API. For each row, the text is passed to the gpt_api function (which should be loaded separately from the provided link) to perform sentiment analysis. The sentiment value returned by the API is extracted and filtered. It should be either "-1" for negative sentiment, "0" for neutral sentiment, or "1" for positive sentiment. If the sentiment value is not within these expected values, it is flagged as an "Invalid number". If there are multiple valid sentiment values, it is flagged as "Multiple valid numbers". Otherwise, it is flagged as "Single valid number". The function constructs a new dataframe (result) containing the original IDs, the extracted sentiment values, and the flags indicating the success of sentiment extraction.

## 08 - whisper_api.R - This script provides two functions, whisper_transcribe and whisper_translate, that utilize the OpenAI API for audio transcription and translation tasks.

Here is a brief description of each function:

*`whisper_transcribe`: Transcribes an audio file. The function makes an HTTP POST request to the OpenAI API endpoint for audio transcription. It sends the audio file, model name, response format, temperature, language (if provided), and prompt (if provided) as the request body. The response from the API is then parsed and returned as a list.
*`whisper_translate`: Translates an audio file. It has similar parameters to the `whisper_transcribe` function, except it does not have the language parameter. Instead, it has the prompt parameter for specifying the translation prompt. Similarly to `whisper_transcribe`, the `whisper_translate` function makes an HTTP POST request to the OpenAI API endpoint for audio translation. It sends the audio file, model name, response format, temperature, and prompt (if provided) as the request body. The response from the API is parsed and returned as a list.

## 09 - `whisper_to_gpt.R` - This script defines a function called whisper_to_gpt that transcribes recorded audio using OpenAI's Whisper API and processes the transcription using OpenAI's GPT API. The function performs the following steps: (1) Records audio for the specified duration using the record function from the audio package. (2) Transcribes the recorded audio using the Whisper API by calling the whisper_transcribe function from the whisper_api.R file. It passes the audio file, API key, language, and prompt as arguments to the function. (3) Checks the status of the transcription and retrieves the resulting transcript. (4) Sends the transcription to the GPT API by calling the gpt_api function from the gpt_api.R file. It passes the transcription as an argument to the function.
(5) Returns the response from the GPT API. 

The script also includes links to the gpt_api.R and whisper_api.R files, which should be loaded in order to use the whisper_to_gpt function


# Contact
- email: napoleonic_bores@proton.me
- discord: elkronos
