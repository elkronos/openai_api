library(httr)
library(stringr)
library(lubridate)

# Set API key as system variable
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}
set_api_key("sk-YOUR-API-KEY")

#' This function sends a prompt to the OpenAI API and returns the generated text. 
#' It handles API communication, error checking, and basic formatting of the response.
#'
#' Register for an API secret here: https://platform.openai.com/account/api-keys
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
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50, 
                    system_message = NULL, num_retries = 3, pause_base = 1, 
                    presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
    # prepend system message to messages list
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }
  
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
  
  clean_message <- gsub("\n", " ", message) # replace newlines with spaces
  clean_message <- str_trim(clean_message) # trim white spaces
  return(clean_message)
}

#' @title Date Parsing Function Using GPT
#' @description This function tries to parse date strings in a dataframe column, first with lubridate and then with a call to GPT for any dates that can't be parsed with lubridate.
#' @param df A data frame that contains the column to be parsed.
#' @param column A character string that specifies the name of the column to be parsed.
#' @param system_message A character string that specifies the system message to be sent to GPT (defaults to a specific prompt for date parsing).
#' @param temperature A numeric value that specifies the randomness of the GPT output (defaults to 0.1).
#' @return A data frame that includes the original date strings, parsed dates, the status of the parsing operation, and who parsed the date (R or GPT).
#' @importFrom lubridate ymd
#' @importFrom stringr str_extract str_remove_all str_trim
#' @examples 
#' @examples 
#' \dontrun{
#' id <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
#' dates <- c("2023-07-01", "July 1st 2023", "01-07-2023", "July 1, 2023", "Jul-2023", "July-2023", "07/01/2023", "01/07/2023",
#'            "The Second of May 2023", "1234-5678", "2000-Feb-30", "APR202023", "2000-FEB-01", "February first twenty twenty-three",
#'            "MAR202315", "12-01-2023")
#' 
#' df <- data.frame(id = id,
#'                  date_column = dates)
#' 
#' result_df <- gpt_date_parser(df, "date_column")
#' 
#' print(result_df)
#' }
#' @export
gpt_date_parser <- function(df, column, system_message = "You are an expert date identifier. You will be given text which you must read and determine if it's a real date. If it is a real date, restate it in YYYY-MM-DD format, surrounded with '#'. If there is no date, state that there is no date.", temperature = 0.1) {
  
  parsed_df <- data.frame(id = integer(), original = character(), parsed = character(), status = character(), parsed_by = character(), stringsAsFactors = FALSE)
  failed_df <- data.frame(id = integer(), original = character(), parsed = character(), status = character(), parsed_by = character(), stringsAsFactors = FALSE)
  
  for (i in 1:nrow(df)) {
    id <- df$id[i]
    date <- df[[column]][i]
    parsed_date <- tryCatch(
      ymd(date),
      error = function(e) return(NA)
    )
    
    if (!is.na(parsed_date)) {
      parsed_df <- rbind(parsed_df, data.frame(id = id, original = date, parsed = as.character(parsed_date), status = 'parsed', parsed_by = 'R', stringsAsFactors = FALSE))
      print(paste("Parsed:", date))
    } else {
      failed_df <- rbind(failed_df, data.frame(id = id, original = date, parsed = NA, status = 'failed', parsed_by = 'NA', stringsAsFactors = FALSE))
      print(paste("Failed to parse:", date))
    }
  }
  
  for (i in 1:nrow(failed_df)) {
    original_date <- failed_df$original[i]
    prompt = paste0("What date is specified in '", original_date, "'?")
    
    response <- gpt_api(prompt, system_message = system_message, temperature = temperature)
    response_date <- str_extract(response, "#.*#")
    
    if (!is.na(response_date)) {
      response_date <- str_remove_all(response_date, "#")
      response_date <- str_trim(response_date)
      failed_df$parsed_by[i] <- 'GPT'
    } else {
      response_date <- NA
      failed_df$parsed_by[i] <- 'NA'
    }
    
    failed_df$parsed[i] <- response_date
    print(paste("Response from GPT:", response_date))
  }
  
  result_df <- rbind(parsed_df, failed_df)
  result_df <- result_df[order(result_df$id), ]
  return(result_df)
}