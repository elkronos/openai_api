# Load packages
library(httr)
library(tools)
library(pdftools)
library(stringr)
library(readtext)
library(tesseract)
library(magrittr)
library(magick)
# Set API key. 
# Sign-up for API key with a plus account here: https://platform.openai.com/signup
# Get key once signed up here: https://platform.openai.com/account/api-keys
api_key <- "sk-your-key-here"  # Replace with your actual API key

#' Parse documents to tokens for GPT
#'
#' This function parses different types of documents (PDF, DOCX, TXT) and images (PNG, JPG, TIF), performs optical character recognition (OCR) on scanned documents and images, and converts them into tokens for further processing with GPT (Generative Pre-trained Transformer) models. It removes whitespace, special characters, and numbers from the text, and splits it into chunks of tokens to avoid exceeding the model's token limit.
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
#' @import magick
#' @import tesseract
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
#' \code{\link{pdftools::pdf_text}}, \code{\link{readtext::readtext}}, \code{\link{readLines}}, \code{\link{magick::image_read_pdf}}, \code{\link{tesseract::ocr}}
#' 
#' @export
parse_text <- function(file, remove_whitespace = TRUE, remove_special_chars = TRUE, remove_numbers = TRUE) {
  
  ext <- tools::file_ext(file)
  
  text <- switch(
    ext,
    pdf = {
      # Check first if it's a text-based PDF or scanned PDF
      text_check <- pdftools::pdf_text(file)
      if(length(text_check) > 0 && nchar(text_check[[1]]) > 0) {
        # It's a text-based PDF, use normal text extraction
        pdftools::pdf_text(file)
      } else {
        # It's a scanned PDF, convert to image and then use OCR
        image <- magick::image_read_pdf(file)
        ocr(image)
      }
    },
    docx = readtext::readtext(file),
    txt = readLines(file),
    png = ocr(file),
    jpg = ocr(file),
    tif = ocr(file),
    stop("Unsupported file type.")
  )
  
  text_combined <- paste(text[nchar(text) > 0], collapse = " ")
  text_combined <- stringr::str_trim(text_combined) %>% 
    stringr::str_remove_all("[^[:alnum:]\\s]") %>% 
    stringr::str_remove_all("\\d+")
  
  tokens <- unlist(strsplit(text_combined, "\\s+"))
  token_chunks <- split(tokens, ceiling(seq_along(tokens)/15000))
  
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
#' This function reads a list of text chunks and a question, and uses the GPT model to generate a response
#' based on the text and question. The function sends requests to the OpenAI API and processes the responses. This enhanced version features better error handling, question logging, improved content editing, and parameter tuning.
#'
#' @param chunk_list A list of text chunks to process.
#' @param question The question to ask the model.
#' @param model The model to use. Default is "gpt-3.5-turbo-1106". See https://platform.openai.com/docs/models/gpt-3-5
#' @param temperature The temperature parameter for text generation. Default is 0.0.
#' @param max_tokens The maximum number of tokens in the generated response. Default is 2500.
#' @param system_message_1 The initial system message to be included in the conversation with the model. Default is "You are a research assistant trying to answer questions posed based on text you are supplied with..."
#' @param system_message_2 The system message for the content editor stage. Default is "You are a content editor who will read the previous responses..."
#' @param num_retries The number of times to retry the API request in case of failure. Default is 5.
#' @param pause_base The base pause time between retries. Default is 3.
#' @param presence_penalty The presence penalty for text generation. Default is 0.0.
#' @param frequency_penalty The frequency penalty for text generation. Default is 0.0.
#'
#' @return The generated response from the GPT model.
#'
#' @importFrom httr POST add_headers content_type_json encode RETRY stop_for_status
#' @importFrom stringr str_trim
#' @importFrom base Sys.time cat
#'
#' @examples
#' # Set the path to your file
#' file <- "C:/path/Brants et al. 2007. Large language models in machine translations.pdf"
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
#'
#' # Access question log
#' log_content <- readLines("user_questions_GPT_answers.log")
#' print(log_content)
#' 
#' @export
gpt_read <- function(chunk_list, question = NULL, model = "gpt-3.5-turbo-1106", temperature = 0.0, max_tokens = 16000, 
                     system_message_1 = "You are a research assistant trying to answer questions posed based on text you are supplied with. Your goal is to provide answers based on the text provided. If the question is not related to or answered by the text, please only say you cannot find the answer in the text.",
                     system_message_2 = "You are a content editor who will read the previous responses from the AI and merge them into a single concise response to the question. If they mention that the answer cannot be found in the text for each chunk of text, only say that.",
                     num_retries = 5, pause_base = 3, delay_between_chunks = 2, presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  if (is.null(question)) {
    stop("A question must be provided.")
  }
  
  results <- list()
  
  for (chunk in chunk_list) {
    messages <- list(
      list(role = "system", content = system_message_1),
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
    
    response <- try(
      RETRY(
        "POST",
        url = "https://api.openai.com/v1/chat/completions", 
        add_headers(Authorization = paste("Bearer", api_key)),
        content_type_json(),
        encode = "json",
        times = num_retries,
        pause_base = pause_base,
        body = body_data
      ),
      silent = TRUE
    )
    
    if (inherits(response, "try-error")) {
      print(paste("Error in request:", response))
      next
    }
    
    stop_for_status(response)
    
    response_content <- content(response)
    
    if (length(response_content$choices) > 0) {
      message <- response_content$choices[[1]]$message$content
    } else {
      message <- "The model did not return a message. You may need to increase max_tokens."
    }
    
    clean_message <- gsub("\n", " ", message) 
    clean_message <- str_trim(clean_message) 
    results <- append(results, list(clean_message))
    
    # Add a pause between processing each chunk
    Sys.sleep(delay_between_chunks)
    
    system_message <- system_message_2
  }
  
  if (all(sapply(results, function(x) grepl("not found|irrelevant|applicable", tolower(x))))) {
    final_answer = "The answer to the question was not found in the supplied text."
  } else {
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
    
    final_answer = gsub("\n", " ", message) 
    final_answer = str_trim(final_answer) 
  }
  
  log_entry <- paste(Sys.time(), "\t", question, "\t", final_answer, "\n")
  cat(log_entry, file = "user_questions_GPT_answers.log", append = TRUE)
  
  return(final_answer)
}