# openai_api
Interact with openai APIs using (DALLE2, GPT, Whisper). The code base is primairly in R, but refactored scripts are being developed and tested. For best results, use GPT 4.

## Scripts

Below is a summary of each directory. Please see specific scripts for detailed roxygen about their uses with examples.

### 01 - api_base - Basic API connectivity.
* `dalle_api.R` - Sends prompt to generate and download images from DALLE.
* `fine_tune_model.R` -  Function to fine-tune a GPT model using the OpenAI API.
* `gpt_api.R` - Sends a prompt to GPT via the OpenAI API and returns the generated text.
* `gpt_contextual_api.R` - Sends a prompt to GPT via the API; and affords the ability to manage the context of a conversation and keep multiple sessions.
* `whisper_api.R` - Transcribes audio files.

### 02 - assistants - Scripts designed to assist in completeing specific tasks.
* `gpt_audioquery.R` - Record audio for a given duration, transcribe it using the Whisper API, and send the transcription to the GPT.
* `gpt_dateparse.R` - Parse dates with GPT.
* `gpt_classifier.R` - Classifies the type of data in each column in a dataset.
* `gpt_finetune.R` - Wrapper function for fine-tuning models using your own documents. Provides functions to upload documents, check status, and cancel jobs.
* `gpt_gridsearch.R` - Performs a grid search over the specified GPT models and parameters, using a given prompt.
* `gpt_nlp.R` - Function that allows the user to convert text into root words, label the part of speech (noun, verb, adverb, etc), or analyze the sentiment of text (-1, 0, 1).
* `gpt_persona.R` - Illustrates how to vary system messages (change roles/personas) across a dataset to generate multiple responses.
* `gpt_random_number_experiment.R` - Based on the `gpt_gridsearch.R` assistant, allows use to vary prompts, temperature, system messages, and add context. Designed to perform an experiment whereby GPT picks random numbers between 0 - 10.
* `gpt_read.R` -  Reads a list of text chunks and a question, and uses the GPT model to generate a response based on the text.
* `gpt_sentiment.R` - Conducts sentiment analysis on a given dataframe using the GPT.

### 03 - dashboards - Interactive dashboards integrated with an OpenAI API.
* `gpt_sentiment_assistant.R` - Use GPT to analyze sentiment via a Shiny dashboard. Populates barplot and searchable table.

### 04 - refactored - Scripts refactored into other languages.

See a table of scripts and refactored languages below.

| Package Name   | Description                                                  | Python      | Ruby        |
|----------------|--------------------------------------------------------------|-------------|-------------|
| gpt_api        | Call the GPT API.                     | &#x2713;    | &#x2713;    |
| gpt_contextual_api        | Call the GPT API.                     |        | &#x2713;    |
| gpt_classifier | Classify/label data using GPT.        | &#x2713;    |             |
| gpt_dateparser | Parse dates using GPT.      | &#x2713;    | &#x2713;    |
| gpt_read       | Read documents and ask questions. | &#x2713; |             |
| gpt_sentiment  | Code sentiment of text using GPT.     | &#x2713;    | &#x2713;    |
| whisper_api    | Translate or transcribe audio using GPT. | &#x2713; |             |
