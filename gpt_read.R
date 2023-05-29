# Load packages
library(httr)
library(tools)
library(pdftools)
library(stringr)
library(readtext)
# Set API key. 
# Sign-up for API key with a plus account here: https://platform.openai.com/signup
# Get key once signed up here: https://platform.openai.com/account/api-keys
api_key <- "sk-your_api_key"  # Replace with your actual API key

#' Parse documents to tokens for GPT
#'
#' This function parses different types of documents (PDF, DOCX, TXT) and converts them into tokens for further processing with GPT (Generative Pre-trained Transformer) models. It removes whitespace, special characters, and numbers from the text, and splits it into chunks of tokens to avoid exceeding the model's token limit.
#'
#' @param file A character string specifying the path to the input file.
#' @param remove_whitespace Logical indicating whether to remove leading and trailing whitespace from the text. Default is \code{TRUE}.
#' @param remove_special_chars Logical indicating whether to remove special characters from the text. Default is \code{TRUE}.
#' @param remove_numbers Logical indicating whether to remove numbers from the text. Default is \code{TRUE}.
#'
#' @return A list of character vectors, where each vector represents a chunk of tokens from the input document.
#'
#' @import httr
#' @import tools
#' @import pdftools
#' @import stringr
#' @import readtext
#'
#' @examples
#' # Example 1: Parsing a PDF document
#' pdf_file <- "path/to/document.pdf"
#' parsed_tokens <- parse_text(file = pdf_file)
#' 
#' # Example 2: Parsing a DOCX document
#' docx_file <- "path/to/document.docx"
#' parsed_tokens <- parse_text(file = docx_file)
#' 
#' # Example 3: Parsing a TXT document
#' txt_file <- "path/to/document.txt"
#' parsed_tokens <- parse_text(file = txt_file)
#'
#' @seealso
#' \code{\link{pdftools::pdf_text}}, \code{\link{readtext::readtext}}, \code{\link{readLines}}
#' 
#' @export
parse_text <- function(file, remove_whitespace = TRUE, remove_special_chars = TRUE, remove_numbers = TRUE) {
  
  ext <- tools::file_ext(file)
  
  text <- switch(
    ext,
    pdf = pdftools::pdf_text(file),
    docx = readtext::readtext(file),
    txt = readLines(file),
    stop("Unsupported file type.")
  )
  
  text_combined <- paste(text[nchar(text) > 0], collapse = " ")
  text_combined <- stringr::str_trim(text_combined) %>% 
    stringr::str_remove_all("[^[:alnum:]\\s]") %>% 
    stringr::str_remove_all("\\d+")
  
  tokens <- unlist(strsplit(text_combined, "\\s+"))
  token_chunks <- split(tokens, ceiling(seq_along(tokens)/3000))
  
  # Convert each token chunk into a string
  chunk_list <- lapply(token_chunks, paste, collapse = " ")
  
  return(chunk_list)
}

#' Extract key words and passages
#'
#' This function extracts passages from a given text that contain specified keywords. 
#' It searches for each keyword in the text and returns a list of passages (windows) 
#' that surround the occurrences of the keywords.
#'
#' @param text The input text from which to extract passages.
#' @param keywords A character vector of keywords to search for in the text.
#' @param window_size An integer specifying the size of the window (in characters) 
#'   around each keyword occurrence. The default value is 500.
#'
#' @return A list of passages (windows) containing the occurrences of the keywords. 
#'   Each element in the list represents a passage.
#'
#' @importFrom base paste
#' @importFrom base nchar
#' @importFrom base append
#' @importFrom base substr
#' @importFrom base min
#' @importFrom base max
#' @importFrom base as.vector
#' @importFrom base gregexpr
#' @importFrom base collapse
#' @importFrom base ignore.case
#'
#' @examples
#' text <- "This is a sample text containing some keywords. The keywords are important for analysis."
#' keywords <- c("sample", "important")
#' search_text(text, keywords)
#'
#' @seealso
#' Other functions: \code{\link{findOccurrences}}, \code{\link{extractKeywords}}
search_text <- function(text, keywords, window_size = 500) {
  result <- list()
  
  keywords <- as.vector(keywords)
  text_str <- paste(text, collapse = " ")
  
  for (keyword in keywords) {
    keyword_indices <- gregexpr(keyword, text_str, ignore.case = TRUE)[[1]]
    keyword_indices <- keyword_indices[keyword_indices != -1]
    
    for (index in keyword_indices) {
      window <- substr(text_str, max(1, index - window_size), min(nchar(text_str), index + window_size))
      result <- append(result, list(window))
    }
  }
  
  return(result)
}

#' Read and process text using GPT
#'
#' This function reads a list of text chunks and a question, and uses the GPT-3.5 Turbo model to generate a response
#' based on the text and question. The function sends requests to the OpenAI API and processes the responses.
#'
#' @param chunk_list A list of text chunks to process.
#' @param question The question to ask the model.
#' @param model The model to use. Default is "gpt-3.5-turbo".
#' @param temperature The temperature parameter for text generation. Default is 0.2.
#' @param max_tokens The maximum number of tokens in the generated response. Default is 100.
#' @param system_message_1 The initial system message to be included in the conversation with the model.
#'                         Default is "You are a research assistant trying to answer questions posed based on text you are supplied with..."
#' @param system_message_2 The system message for the content editor stage. Default is "You are a content editor who will read the previous responses..."
#' @param num_retries The number of times to retry the API request in case of failure. Default is 5.
#' @param pause_base The base pause time between retries. Default is 3.
#' @param presence_penalty The presence penalty for text generation. Default is 0.0.
#' @param frequency_penalty The frequency penalty for text generation. Default is 0.0.
#'
#' @return The generated response from the GPT-3.5 Turbo model.
#'
#' @importFrom httr POST add_headers content_type_json encode
#' @importFrom stringr str_trim
#' @import RETRY
#'
#' @examples
#' # Set the path to your file
#' file <- "C:/Users/JChas/OneDrive/Desktop/pdf_examples/Brants et al. 2007. Large language models in machine translations.pdf"
#'
#' # Call the function
#' text <- parse_text(file)
#'
#' # Ask question
#' question <- "what is this article about?"
#' # Get answer
#' gpt_read(chunk_list = text, question = question) -> response_1
#' # Review response
#' print(response_1)
#'
#' question <- "what is the stupid backoff method?"
#' gpt_read(chunk_list = text, question = question) -> response_2
#' print(response_2)
#'
#' question <- "Why do kittens meow?"
#' gpt_read(chunk_list = text, question = question) -> response_3
#' print(response_3)
gpt_read <- function(chunk_list, question = NULL, model = "gpt-3.5-turbo", temperature = 0.2, max_tokens = 100, 
                     system_message_1 = "You are a research assistant trying to answer questions posed based on text you are supplied with. Your goal is to provide answers based on the text provided. If the question is not related to or answered by the text, please only say you cannot find the answer in the text.",
                     system_message_2 = "You are a content editor who will read the previous responses from the AI and merge them into a single concise response to the question. If they mention that the answer cannot be found in the text for each chunk of text, only say that.",
                     num_retries = 5, pause_base = 3, presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  if (is.null(question)) {
    stop("A question must be provided.")
  }
  
  results <- list()
  
  # Initial system message
  system_message <- system_message_1
  
  for (chunk in chunk_list) {
    # Creating messages list for each chunk
    messages <- list(
      list(role = "system", content = system_message),
      list(role = "user", content = chunk),
      list(role = "user", content = question)
    )
    
    body_data <- list(
      model = model,
      temperature = temperature,
      max_tokens = max_tokens,
      messages = messages,
      presence_penalty = presence_penalty,
      frequency_penalty = frequency_penalty
    )
    
    # Print debugging info
    print(paste("Sending request with body data:", toString(body_data)))
    
    response <- RETRY(
      "POST",
      url = "https://api.openai.com/v1/chat/completions", 
      add_headers(Authorization = paste("Bearer", api_key)),
      content_type_json(),
      encode = "json",
      times = num_retries,
      pause_base = pause_base,
      body = body_data
    )
    
    stop_for_status(response)
    
    if (length(content(response)$choices) > 0) {
      message <- content(response)$choices[[1]]$message$content
    } else {
      message <- "The model did not return a message. You may need to increase max_tokens."
    }
    
    clean_message <- gsub("\n", " ", message) 
    clean_message <- str_trim(clean_message) 
    results <- append(results, list(clean_message))
    
    # Update system message for the next iteration
    system_message <- system_message_2
  }
  
  # Check if answer was found in the supplied text
  if (all(sapply(results, function(x) grepl("not found|irrelevant|applicable", tolower(x))))) {
    return("The answer to the question was not found in the supplied text.")
  }
  
  # Content editor stage
  messages <- list(
    list(role = "system", content = system_message),
    list(role = "user", content = paste(results, collapse = "\n")),
    list(role = "user", content = question)
  )
  
  body_data <- list(
    model = model,
    temperature = temperature,
    max_tokens = max_tokens,
    messages = messages,
    presence_penalty = presence_penalty,
    frequency_penalty = frequency_penalty
  )
  
  # Print debugging info
  print(paste("Sending request with body data:", toString(body_data)))
  
  response <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/chat/completions", 
    add_headers(Authorization = paste("Bearer", api_key)),
    content_type_json(),
    encode = "json",
    times = num_retries,
    pause_base = pause_base,
    body = body_data
  )
  
  stop_for_status(response)
  
  if (length(content(response)$choices) > 0) {
    message <- content(response)$choices[[1]]$message$content
  } else {
    message <- "The model did not return a message. You may need to increase max_tokens."
  }
  
  clean_message <- gsub("\n", " ", message) 
  clean_message <- str_trim(clean_message) 
  
  return(clean_message)
}