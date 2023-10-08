#' Use GPT for natural language processing
# Load required packages
library(httr)
library(stringr)
library(dplyr)

#' Chunk Text into Smaller Fragments
#'
#' This function takes a text input and divides it into smaller chunks, each
#' containing a maximum of 3000 characters. It is useful when you need to
#' process large texts in smaller, more manageable fragments.
#'
#' @param text A character vector containing the input text that you want to
#'             divide into smaller chunks.
#'
#' @return A list of character vectors, where each element of the list
#'         represents a chunk of text with a maximum of 3000 characters.
#'
#' @import stringr
#'
#' @examples
#' # Example Usage:
#' input_text <- "This is a sample text that we want to chunk into smaller pieces. It should be divided into multiple chunks, each containing a maximum of 3000 characters. This is useful for processing large texts."
#'
#' # Chunk the input text
#' chunked_text <- chunk_text(input_text)
#'
#' # Print the first chunk
#' cat(chunked_text[[1]])
#' 
#' # Print the second chunk
#' cat(chunked_text[[2]])
#'
#' # Print the third chunk (if available)
#' if (length(chunked_text) >= 3) {
#'   cat(chunked_text[[3]])
#' }
#'
#' @export
chunk_text <- function(text) {
  # Tokenize text
  tokens <- unlist(strsplit(text, "\\s+"))
  # Divide tokens into chunks of 3000
  token_chunks <- split(tokens, ceiling(seq_along(tokens)/3000))
  # Combine tokens back into chunks of text
  text_chunks <- lapply(token_chunks, paste, collapse = " ")
  
  return(text_chunks)
}


#' This function sends a prompt to the OpenAI API and returns the generated text. 
#' It handles API communication, error checking, and basic formatting of the response.
#'
#' Register for an API secret here: https://platform.openai.com/account/api-keys
#' 
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}
set_api_key("sk-YOUR-KEY")
print(Sys.getenv("OPENAI_API_KEY"))
#' 
#' @importFrom httr RETRY POST content_type_json add_headers stop_for_status content
#' @importFrom stringr str_trim
#' 
#' @param prompt The prompt to send to the API.
#' @param model The model to use for generating the response. Default is "gpt-3.5-turbo".
#'     This can be adjusted according to the available models in the OpenAI API (such as "gpt-4")
#' @param temperature The temperature to use for generating the response. Default is 0.5.
#' @param max_tokens The maximum number of tokens in the response. Default is 50.
#' @param system_message Optional initial system message to set the behavior of the AI.
#'     Default is NULL.
#' @param num_retries Number of times to retry the request in case of failure. Default is 3.
#' @param pause_base The number of seconds to wait between retries. Default is 1.
#' @param presence_penalty The penalty for new tokens based on their presence in the input. Default is 0.0.
#' @param frequency_penalty The penalty for new tokens based on their frequency in the input. Default is 0.0.
#'
#' @return A character string containing the generated text.
#'
#' @examples
#' gpt_api("Write a poem about my cat Sam")
#' gpt_api("Tell me a joke.", model = "gpt-3.5-turbo")
#' gpt_api("Write a poem about my cat Sam", model = "gpt-4")
#' 
#' @export
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.1, max_tokens = 50, 
                    system_message = NULL, num_retries = 3, pause_base = 1, 
                    presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
    # prepend system message to messages list
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }
  
  # Printing the prompt and system message being sent to GPT API
  cat("Sending the following to GPT API:\n")
  cat("Prompt: ", prompt, "\n")
  if(!is.null(system_message)) cat("System Message: ", system_message, "\n")
  
  response <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/chat/completions", 
    add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY"))),
    content_type_json(),
    encode = "json",
    times = num_retries,
    pause_base = pause_base,
    body = list(
      model = model,
      temperature = temperature,
      max_tokens = max_tokens,
      messages = messages,
      presence_penalty = presence_penalty,
      frequency_penalty = frequency_penalty
    )
  )
  
  # Check for HTTP errors
  stop_for_status(response)
  
  # Get and clean the message
  if (length(content(response)$choices) > 0) {
    message <- content(response)$choices[[1]]$message$content
  } else {
    message <- "The model did not return a message. You may need to increase max_tokens."
  }
  
  # Printing the message received from GPT API
  cat("Received the following from GPT API:\n")
  cat(message, "\n")
  
  clean_message <- gsub("\n", " ", message) # replace newlines with spaces
  clean_message <- str_trim(clean_message) # trim white spaces
  
  return(clean_message)
}


#' Process GPT Output
#'
#' This function processes a list of GPT model outputs based on the specified task type
#' and returns the processed data in a list of data frames.
#'
#' @param output_list A list of character strings, where each string represents
#'                   the output generated by the GPT model.
#' @param task_type   A character string specifying the type of processing to be
#'                   applied to the GPT outputs. Valid values are "root", "sentiment",
#'                   or "label".
#'
#' @return A list of processed data frames. The structure of the data frames depends
#'         on the task type:
#'         - For "root" task_type, the data frame contains columns: 'sequence' and 'root'.
#'         - For "sentiment" task_type, the data frame contains columns: 'sequence', 'word',
#'           'sentiment', and 'label'.
#'         - For "label" task_type, the data frame contains columns: 'sequence', 'word',
#'           'speech', and 'label'.
#'
#' @details
#' The function takes the GPT model outputs, splits them, and processes them differently
#' based on the specified task type. It can handle tasks related to extracting roots,
#' sentiment analysis, and labeling words.
#'
#' @seealso
#' \code{\link{strsplit}}, \code{\link{gsub}}, \code{\link{data.frame}}
#'
#' @importFrom base seq_along
#'
#' @importFrom utils strsplit
#'
#' @examples
#' \dontrun{
#' # Example 1: Process GPT outputs for root extraction
#' root_outputs <- c("The quick brown fox", "jumps over the lazy dog")
#' processed_data <- process_gpt_output(root_outputs, task_type = "root")
#'
#' # Example 2: Process GPT outputs for sentiment analysis
#' sentiment_outputs <- c("Happy:2 Sad:0", "Excited:1")
#' processed_data <- process_gpt_output(sentiment_outputs, task_type = "sentiment")
#'
#' # Example 3: Process GPT outputs for word labeling
#' label_outputs <- c("Noun:cat Verb:runs Adjective:happy", "Adverb:quick")
#' processed_data <- process_gpt_output(label_outputs, task_type = "label")
#' }
#'
#' @export
process_gpt_output <- function(output_list, task_type){
  processed_list <- lapply(output_list, function(output) {
    
    # For "root" task_type
    if(task_type == "root"){
      words <- unlist(strsplit(output, "\\s+"))
      data.frame(
        sequence = seq_along(words),
        root = words,
        stringsAsFactors = FALSE
      )
      
      # For "sentiment" task_type  
    } else if(task_type == "sentiment"){
      split_output <- strsplit(output, "\\s+")[[1]]
      words <- split_output[seq(1, length(split_output), 2)]
      sentiments <- as.numeric(gsub(":", "", split_output[seq(2, length(split_output), 2)]))
      labels <- c("Negative", "Neutral", "Positive")[sentiments + 2] # Adding 2 to make indices 1, 2, 3
      
      data.frame(
        sequence = seq_along(words),
        word = gsub(":", "", words),
        sentiment = sentiments,
        label = labels,
        stringsAsFactors = FALSE
      )
      
      # For "label" task_type  
    } else if(task_type == "label"){
      words <- unlist(strsplit(output, "\\s+"))
      numeric_labels <- gsub("\\D", "", words) # Remove non-digits
      word_labels <- words
      word_labels[nchar(numeric_labels) > 0] <- gsub("_\\d+", "", word_labels[nchar(numeric_labels) > 0])
      
      labels <- c("Noun", "Pronoun", "Adjective", "Verb", "Adverb", 
                  "Preposition", "Conjunction", "Interjection", "Article")[as.numeric(numeric_labels)]
      
      data.frame(
        sequence = seq_along(words),
        word = word_labels,
        speech = ifelse(nchar(numeric_labels) > 0, numeric_labels, NA),
        label = ifelse(!is.na(labels), labels, "Unknown"),
        stringsAsFactors = FALSE
      )
    } else {
      stop("Invalid task type. Select from 'root', 'label', or 'sentiment'.")
    }
  })
  
  return(processed_list)
}


#' Perform Natural Language Processing using GPT-3
#'
#' This function interacts with the GPT-3 language model to perform various NLP tasks
#' such as labeling words, extracting root forms, or assigning sentiment scores to words
#' within provided text chunks.
#'
#' @param text_chunks A character vector containing the text chunks to be processed.
#' @param task A character string specifying the NLP task to perform. Options are "label"
#'   (default), "root", or "sentiment".
#'
#' @return A list of processed text chunks according to the specified task.
#'
#' @details
#' This function communicates with the GPT-3 model via the `gpt_api` function to process
#' the input text chunks. Depending on the specified task, it can append labels, extract
#' root forms, or assign sentiment scores to words within the text.
#'
#' @references
#' You need to have the `gpt_api` function available and properly configured to use this
#' function. Please refer to the documentation of `gpt_api` for more information.
#'
#' @examples
#' \dontrun{
#' # Example usage for labeling words in a text chunk
#' text_chunks <- c("The quick brown fox jumps", "over the lazy dog.")
#' labeled_text <- gpt_nlp(text_chunks, task = "label")
#' }
#'
#' @importFrom your_package gpt_api process_gpt_output
#'
#' @export
gpt_nlp <- function(text_chunks, task = "label") {
  system_message <- switch(task,
                           label = "Append every word in the following text with an underscore followed by a numeric value. Only append a _1 for noun, _2 for pronoun, _3 for adjective, _4 for verb, _5 for adverb, _6 for preposition, _7 for conjunction, _8 for interjection, and _9 for article. If the word is not used as one of those, please omit a label.",
                           root = "Transform all words in the subsequent text to their root form by removing any prefixes or suffixes and placing the root word.",
                           sentiment = "Assign a sentiment score to each word in the provided text. Utilize _-1 for negative_, _0 for neutral_, and _1 for positive_ sentiment. If a word does not possess a clear sentiment, please omit a label.",
                           stop("Invalid task. Select from 'label', 'root', or 'sentiment'.")
  )
  
  labeled_text_chunks <- lapply(text_chunks, function(chunk) {
    gpt_api(
      prompt = chunk,
      system_message = system_message,
      max_tokens = 3000
    )
  })
  
  # Automatically call process_gpt_output using the received labeled_text_chunks and the task as task_type
  processed_output <- process_gpt_output(labeled_text_chunks, task)
  
  return(processed_output)
}


#' \dontrun{
#' # Example usage with fake text data
#' text_input <- "Once upon a time, in a quaint little village, there lived a curious little cat named Whiskers. Whiskers was not your average cat. He was adventurous, always seeking new mysteries to solve and places to explore. His small size and agile body made it easy for him to sneak into places unnoticed, providing him a secret gateway into the worlds unknown to many.
#' 
#' Every morning, he would wander through the streets of the village, exploring the various shops and alleyways. He was particularly fond of the fish market, where the scent of fresh fish wafted through the air, teasing his senses and enticing his stomach.
#' 
#' One day, while exploring a new alleyway, Whiskers encountered a strange-looking box. Intrigued, he approached it cautiously, sniffing around to detect any unfamiliar scents. Suddenly, the box wobbled, startling Whiskers. He arched his back and puffed up his fur, ready to defend himself from any possible threat. To his surprise, out popped a tiny mouse, looking just as startled as Whiskers.
#' 
#' The mouse squeaked, 'Oh dear! You gave me quite a fright!' Whiskers, realizing that the mouse was not a threat, relaxed and curiously inquired, 'Who are you, little one?'
#' 
#' The mouse introduced herself as Minny and explained that she had accidentally trapped herself inside the box while searching for food. Whiskers, being the adventurous and kind-hearted cat he was, decided to help Minny find a safer place to live, away from the bustling streets and potential dangers of the village.
#' 
#' The unlikely duo, Whiskers and Minny, embarked on a journey through the village, exploring various potential homes for Minny. Throughout their adventure, they encountered numerous obstacles, from navigating through busy streets to avoiding the menacing claws of other, not-so-friendly cats.
#' 
#' Despite their differences and the natural predator-prey relationship, Whiskers and Minny developed an unexpected friendship. They learned that by working together, they could overcome any challenge that came their way. Whiskers realized that not all mice were meant to be chased, and Minny learned that not all cats were to be feared.
#' 
#' In the end, they found a cozy little nook in an old, abandoned house where Minny could live safely. Whiskers promised to visit her regularly, and in return, Minny shared her found seeds and nuts, which Whiskers found peculiar yet oddly tasty.
#' 
#' Their friendship blossomed, proving that kindness and understanding could bridge the gap between the most unlikely of friends, turning a potential enemy into a beloved companion. And so, in the quaint little village, the tales of Whiskers the cat and Minny the mouse spread, teaching everyone that friendships could be found in the most unexpected of places.
#' The end."
#' 
#' # Chunk text
#' chunk_text(text_input) -> chunked_output
#' 
#' # For root task
#' root_processed <- gpt_nlp(chunked_output, task = "root")
#' 
#' # For sentiment task
#' sentiment_processed <- gpt_nlp(chunked_output, task = "sentiment")
#' 
#' # For label task
#' label_processed <- gpt_nlp(chunked_output, task = "label")
#' }