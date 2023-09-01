# Load required libaries
library(data.table)
library(httr)
library(stringr)

# Set API key as system variable
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}
set_api_key("sk-YOUR-API-KEY")


#' Reshape a data frame or a data table into a new data table format
#'
#' This function takes a data frame or a data table and reshapes it into a new data table. The reshaped table has two columns: 'variables', 
#' which contains the names of the original columns, and 'responses', which contains the first 'n' non-NA values from the original columns as a comma-separated string.
#' This function depends on the 'data.table' package.
#'
#' @param dt A data frame or a data table to be reshaped. 
#' @param n A positive integer. This specifies the number of non-NA values to be taken from each column of the input data frame or data table. 
#' If a column contains fewer than 'n' non-NA values, all non-NA values are taken.
#'
#' @return A data table with two columns: 'variables' and 'responses'. 'variables' contains the names of the original columns,
#' and 'responses' contains the first 'n' non-NA values from the original columns as a comma-separated string.
#'
#' @importFrom data.table data.table is.data.table as.data.table rbind
#'
#' @examples
#' # Synthetic data set
#' df <- data.frame(
#'  VAR_1 = c("A", "A", "B", "B", "C", "C", "D", "D"),
#'  VAR_2 = c(1, 2, 5, 4, 5, 7, 8, 8),
#'  VAR_3 = as.Date(c("1999-01-01", "1999-01-02", "1999-01-09", "1999-01-15", "1999-02-01", "1999-02-01", "1999-02-01", "1999-02-01")),
#'  VAR_4 = c("Keanu Reeves", "Patrick Stewart", "Fred Rogers", "Sir David Attenborough", "Betty White", "Tom Hanks", "LeVar Burton", "Jackie Chan"),
#'  VAR_5 = as.factor(c("cat", "dog", NA, "cat", "dog", "dog", "dog", "cat")),
#'  VAR_6 = c(TRUE, FALSE, TRUE, NA, FALSE, TRUE, FALSE, TRUE),
#'  VAR_7 = c("37.7749 N, 122.4194 W", "40.7128 N, 74.0060 W", "51.5072 N, 0.1276 W", NA, "34.0522 N, 118.2437 W", "48.8566 N, 2.3522 E", NA, NA) 
#' )
#'
#' # Use the function to reshape the data frame
#' reshape_dt(df, 2)
reshape_dt <- function(dt, n) {
  
  # Check that n is a positive integer
  if (!is.numeric(n) | n < 1 | round(n) != n) {
    stop("Error: n must be a positive integer.")
  }
  
  # Convert data frame to data.table
  if (!is.data.table(dt)) {
    dt <- as.data.table(dt)
  }
  
  # Create a new data table to store the reshaped data
  new_dt <- data.table(variables = character(), responses = character())
  
  # Iterate over the columns of the original data table
  for (var in names(dt)) {
    
    # Get the column data, remove NA values, and select the first n values
    responses <- dt[[var]]
    responses <- responses[!is.na(responses)]
    responses <- responses[1:min(n, length(responses))]
    
    # Convert the responses to a comma-separated string
    responses_str <- toString(responses)
    
    # Add the variable and its responses to the new data table
    new_dt <- rbind(new_dt, data.table(variables = var, responses = responses_str))
  }
  
  # Return the new data table
  return(new_dt)
}

#' Make API call to OpenAI GPT
#'
#' This function sends a POST request to the OpenAI API endpoint and returns the AI generated text.
#'
#' @param prompt The input message for the AI.
#' @param model The model to be used by the AI. Default is "gpt-3.5-turbo".
#' @param temperature The temperature parameter for the AI. Controls the randomness of the AI's output. Default is 0.5.
#' @param max_tokens The maximum number of tokens for the AI to generate. Default is 50.
#' @param system_message An optional system message to guide the AI.
#' @param num_retries The number of retry attempts for the API call. Default is 3.
#' @param pause_base The base of the exponential back-off for retrying API call. Default is 1.
#' @param presence_penalty The penalty for new token generation. Default is 0.0.
#' @param frequency_penalty The penalty for frequent token generation. Default is 0.0.
#' 
#' @return A clean AI generated message.
#' 
#' @importFrom httr RETRY add_headers stop_for_status content
#' @importFrom jsonlite toJSON
#' @importFrom stringr str_trim
#' @importFrom glue glue
#' @importFrom purrr possibly
#' 
#' @export
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50, 
                    system_message = NULL, num_retries = 3, pause_base = 1, 
                    presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
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

#' Classify column data types using GPT
#'
#' This function uses OpenAI's GPT to classify the data type of each column in a data frame or data table.
#'
#' @param dt A data frame or data table to classify its columns' data types.
#' @param n The number of initial non-NA values to consider from each column for classification.
#' @param system_message An optional system message to guide the AI.
#' @param temperature The temperature parameter for the AI. Controls the randomness of the AI's output. Default is 0.2.
#'
#' @return A data frame with the names of the original columns, the initial 'n' non-NA values from those columns, the AI's responses,
#' the data types determined by the AI, and a flag indicating whether there were multiple valid numbers in the AI's responses.
#'
#' @importFrom stringr str_extract_all
#' @importFrom purrr possibly
#' 
#' @examples
#' # Synthetic data set
#' df <- data.frame(
#'  VAR_1 = c("A", "A", "B", "B", "C", "C", "D", "D"),
#'  VAR_2 = c(1, 2, 5, 4, 5, 7, 8, 8),
#'  VAR_3 = as.Date(c("1999-01-01", "1999-01-02", "1999-01-09", "1999-01-15", "1999-02-01", "1999-02-01", "1999-02-01", "1999-02-01")),
#'  VAR_4 = c("Keanu Reeves", "Patrick Stewart", "Fred Rogers", "Sir David Attenborough", "Betty White", "Tom Hanks", "LeVar Burton", "Jackie Chan"),
#'  VAR_5 = as.factor(c("cat", "dog", NA, "cat", "dog", "dog", "dog", "cat")),
#'  VAR_6 = c(TRUE, FALSE, TRUE, NA, FALSE, TRUE, FALSE, TRUE),
#'  VAR_7 = c("37.7749 N, 122.4194 W", "40.7128 N, 74.0060 W", "51.5072 N, 0.1276 W", NA, "34.0522 N, 118.2437 W", "48.8566 N, 2.3522 E", NA, NA)
#' )
#'
#' # Classify the data types of the columns
#' gpt_classifier(df, 7)
#'
#' @export
gpt_classifier <- function(dt, n, system_message = "You are tasked with classifying data types. Please review the given series of values and select the most appropriate data type, responding only with the number that most accurately reflects the type of data: [1] Numeric (All unique numerical values), [2] Categorical (Low uniqueness, used mainly as categories e.g. 'high' or 'low'), [3] Character (Words, names, or any string of letters), [4] Date (Any format e.g. 'January 1st', 'Nov-1999', '11-01-1999'), [5] Logical (Only TRUE/FALSE or T/F values), [6] Geospatial (Coordinates). If unsure, select the classification that offers the most analytical flexibility.", temperature = 0.2) {
  
  # reshape the data using reshape_dt function
  dt <- reshape_dt(dt, n)
  
  if (nrow(dt) < 1 || ncol(dt) < 2) {
    stop("The data frame must have at least one row and two columns")
  }
  
  dtype_list <- vector("list", length = nrow(dt))
  flags <- vector("character", length = nrow(dt))
  gpt_responses <- vector("character", length = nrow(dt))
  
  for (i in 1:nrow(dt)) {
    dtype <- gpt_api(as.character(dt[,2][i]), 
                     system_message = system_message, 
                     model = "gpt-3.5-turbo", 
                     temperature = temperature, 
                     max_tokens = 2000, 
                     num_retries = 3, 
                     pause_base = 1, 
                     presence_penalty = 0.0, 
                     frequency_penalty = 0.0)
    
    gpt_responses[i] <- dtype
    
    dtype_numbers <- as.integer(str_extract_all(dtype, "1|2|3|4|5|6")[[1]])
    
    if (length(dtype_numbers) == 0) {
      flags[i] <- "Invalid number"
      dtype_list[[i]] <- NA
    } else {
      dtype_list[[i]] <- dtype_numbers[1]
      flags[i] <- ifelse(length(dtype_numbers) > 1, "Multiple valid numbers", "Single valid number")
    }
  }
  
  dtype_labels <- c('numeric', 'categorical', 'character', 'date', 'logical', 'geospatial')
  DataTypes <- dtype_labels[unlist(dtype_list)]
  
  result <- data.frame(Variables = dt$variables, Responses = dt$responses, GptResponses = gpt_responses, DataType = DataTypes, Flag = flags)
  
  return(result)
}