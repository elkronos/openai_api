# Load necessary packages
library(httr)           # For handling HTTP requests and retries
library(jsonlite)       # For JSON handling
library(stringr)        # For string manipulation
library(purrr)          # For functional programming tools
library(readr)          # For reading and writing data
library(dplyr)          # For data manipulation

#' Log Messages to a File
#'
#' Logs messages to a file and optionally prints them to the console. This is useful for tracking the progress and debugging.
#'
#' @param message Character. The message to be logged.
#' @param log_file Character. The path to the log file.
#' @param verbose Logical. If TRUE, the message will also be printed to the console. Default is TRUE.
#'
#' @return None. The function is used for its side effects.
#'
#' @examples
#' log_message("This is a test message.", "log.txt", verbose = TRUE)
#'
#' @export
log_message <- function(message, log_file, verbose = TRUE) {
  if (verbose) {
    cat(message, "\n")
    write(message, file = log_file, append = TRUE)
  }
}

#' Query GPT API
#'
#' Calls the OpenAI GPT API and retrieves a response based on the provided parameters. It includes error handling and retry logic.
#'
#' @param model Character. The identifier of the language model to use.
#' @param temperature Numeric. Controls randomness in the model's output. Range: 0.0 to 1.0.
#' @param max_tokens Integer. The maximum number of tokens to be generated by the model.
#' @param messages List. A list of messages to be passed to the model, where each message contains 'role' and 'content'.
#' @param num_retries Integer. The number of retry attempts in case of a network error. Default is 3.
#' @param pause_base Numeric. The base of the exponential backoff for retries. Default is 1.
#' @param verbose Logical. If TRUE, log messages will be printed. Default is TRUE.
#' @param log_file Character. The path to the log file. Default is "gpt_log.txt".
#'
#' @importFrom httr RETRY add_headers content_type_json stop_for_status content
#' @importFrom stringr str_trim
#'
#' @return Character. The content of the response from the GPT model, cleaned of newlines and extra spaces.
#'
#' @examples
#' \dontrun{
#' messages <- list(list(role = "user", content = "What is the weather today?"))
#' query_gpt_api("gpt-3.5-turbo", 0.7, 100, messages)
#' }
#'
#' @note Ensure that you have set the `openai_api_key` before using this function.
#'
#' @export
query_gpt_api <- function(model, temperature, max_tokens, messages, num_retries = 3, pause_base = 1, verbose = TRUE, log_file = "gpt_log.txt") {
  tryCatch({
    response <- RETRY(
      "POST",
      url = "https://api.openai.com/v1/chat/completions", 
      add_headers(Authorization = paste("Bearer", openai_api_key)),
      content_type_json(),
      encode = "json",
      times = num_retries,
      pause_base = pause_base,
      body = list(
        model = model,
        temperature = temperature,
        max_tokens = max_tokens,
        messages = messages
      )
    )
    
    stop_for_status(response)
    
    if (length(content(response)$choices) > 0) {
      message_content <- content(response)$choices[[1]]$message$content
    } else {
      message_content <- NULL
    }
    
    clean_message <- gsub("\n", " ", message_content) 
    clean_message <- str_trim(clean_message)
    
    return(clean_message)
  }, error = function(e) {
    log_message(paste("Error in API call: ", e), log_file, verbose)
    return(NULL)
  })
}

#' Extract a Valid Number from GPT Response
#'
#' Parses the response from the GPT API to extract a valid number between 0 and 10.
#'
#' @param response Character. The response from the GPT API.
#' @param verbose Logical. If TRUE, log messages will be printed. Default is TRUE.
#' @param log_file Character. The path to the log file. Default is "gpt_log.txt".
#'
#' @importFrom stringr str_extract
#'
#' @return Integer or NULL. The first valid number between 0 and 10 extracted from the response, or NULL if no valid number is found.
#'
#' @examples
#' extract_valid_number("The number is 7.")
#'
#' @export
extract_valid_number <- function(response, verbose = TRUE, log_file = "gpt_log.txt") {
  if (is.null(response) || response == "") {
    return(NULL)  # If the response is NULL or empty, return NULL
  }
  
  # Extract the first number between 0 and 10
  extracted_number <- str_extract(response, "\\b([0-9]|10)\\b")
  
  if (!is.na(extracted_number)) {
    result <- as.integer(extracted_number)  # Convert to integer
    return(result)
  } else {
    return(NULL)  # Return NULL if no valid number is found
  }
}

#' Generate GPT Response
#'
#' Interacts with the GPT model using a given prompt and optional system message, returning the generated response.
#'
#' @param prompt Character. The user prompt to be processed by the GPT model.
#' @param model Character. The identifier of the language model to use.
#' @param temperature Numeric. Controls randomness in the model's output. Range: 0.0 to 1.0.
#' @param max_tokens Integer. The maximum number of tokens to be generated by the model.
#' @param system_message Character. Optional. A system message to set the behavior or context of the model.
#' @param verbose Logical. If TRUE, log messages will be printed. Default is TRUE.
#' @param log_file Character. The path to the log file. Default is "gpt_log.txt".
#'
#' @importFrom purrr append
#'
#' @return Character. The generated response from the GPT model.
#'
#' @examples
#' \dontrun{
#' generate_gpt_response("Tell me a joke.", "gpt-3.5-turbo", 0.7, 100, "You are a comedian.")
#' }
#'
#' @export
generate_gpt_response <- function(prompt, model, temperature, max_tokens, system_message, verbose = TRUE, log_file = "gpt_log.txt") {
  messages <- list(list(role = "user", content = prompt))
  
  if (!is.null(system_message)) {
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }
  
  result <- query_gpt_api(model, temperature, max_tokens, messages, verbose = verbose, log_file = log_file)
  return(result)
}

#' Random number picking experiment
#'
#' Executes a grid search over various GPT models, temperatures, and system messages, logging results and saving them periodically.
#'
#' @param original_prompt Character. The initial prompt to be sent to the GPT model.
#' @param models Character vector. The list of model names to be tested.
#' @param temperatures Numeric vector. The list of temperatures to be tested.
#' @param max_tokens Integer. The maximum number of tokens to be generated by the model.
#' @param system_messages Character vector. The list of system messages to be tested.
#' @param calls_per_combination Integer. The number of API calls to make for each combination of parameters.
#' @param use_context Logical. If TRUE, the context from previous responses is used in subsequent prompts. Default is TRUE.
#' @param save_interval Integer. The interval at which to save the results to disk. Default is 100.
#' @param save_path Character. The file path where the results will be saved. Default is "gpt_search_results.rds".
#' @param verbose Logical. If TRUE, log messages will be printed. Default is TRUE.
#' @param log_file Character. The path to the log file. Default is "gpt_log.txt".
#'
#' @importFrom utils expand.grid
#' @importFrom base Sys.time Sys.sleep
#' @importFrom readr saveRDS
#'
#' @return Data frame. A data frame containing the results of the grid search, including the parameters used and the responses received.
#'
#' @examples
#' \dontrun{
#' gpt_random_number_experiment(
#'   "Pick a number from 0 to 10.", 
#'   models = c("gpt-3.5-turbo", "gpt-4"),
#'   temperatures = seq(0.0, 1.0, by = 0.2),
#'   max_tokens = 50,
#'   system_messages = c("You are a human.", "You are a machine."),
#'   calls_per_combination = 25
#' )
#' }
#'
#' @export
gpt_random_number_experiment <- function(original_prompt, models, temperatures, max_tokens, system_messages, calls_per_combination, use_context = TRUE, save_interval = 100, save_path = "gpt_search_results.rds", verbose = TRUE, log_file = "gpt_log.txt") {
  
  # Create all combinations of models, temperatures, and system messages
  parameter_combinations <- expand.grid(models = models, 
                                        temperatures = temperatures, 
                                        system_messages = system_messages,
                                        stringsAsFactors = FALSE)
  
  # Initialize an empty data frame to store results
  search_results <- data.frame(
    request_id = integer(),
    timestamp = character(), 
    model = character(), 
    temperature = numeric(), 
    system_message = character(), 
    response = integer(),
    used_context = logical(),
    stringsAsFactors = FALSE
  )
  
  request_id <- 1
  total_combinations <- nrow(parameter_combinations)
  total_calls <- total_combinations * calls_per_combination * (if (use_context) 2 else 1)
  call_counter <- 1
  
  for (i in seq_len(total_combinations)) {
    current_params <- parameter_combinations[i, ]
    
    log_message(paste0("Processing combination ", i, " of ", total_combinations, ": ", 
                       "Model = ", current_params$models, ", Temperature = ", current_params$temperatures, 
                       ", System Message = ", current_params$system_messages), log_file, verbose)
    
    for (context in if (use_context) c(FALSE, TRUE) else c(FALSE)) {
      # Instead of a single last_valid_number, maintain a vector of all previously chosen numbers
      previous_numbers <- c()
      
      for (j in seq_len(calls_per_combination)) {
        log_message(paste0("Processing call ", j, " of ", calls_per_combination, " (Overall call ", call_counter, " of ", total_calls, ")"), log_file, verbose)
        log_message(paste("DEBUG: Context =", context, ", Previous numbers =", paste(previous_numbers, collapse=", ")), log_file, verbose)
        
        # If using context and we have previously chosen numbers, add them to the prompt
        if (context && length(previous_numbers) > 0) {
          adjusted_prompt <- paste(
            original_prompt, 
            "You previously chose:", 
            paste(previous_numbers, collapse = ", ")
          )
        } else {
          adjusted_prompt <- original_prompt
        }
        
        log_message(paste("Prompt:", adjusted_prompt), log_file, verbose)
        
        raw_response <- generate_gpt_response(
          prompt = adjusted_prompt, 
          model = current_params$models, 
          temperature = current_params$temperatures,
          max_tokens = max_tokens, 
          system_message = current_params$system_messages,
          verbose = verbose, 
          log_file = log_file
        )
        
        log_message(paste("GPT Response:", raw_response), log_file, verbose)
        
        parsed_response <- extract_valid_number(raw_response, verbose = verbose, log_file = log_file)
        log_message(paste("DEBUG: Parsed response:", parsed_response), log_file, verbose)
        
        # Append the parsed response to previous_numbers if valid
        if (!is.null(parsed_response)) {
          previous_numbers <- c(previous_numbers, parsed_response)
          log_message(paste("DEBUG: Updated previous numbers:", paste(previous_numbers, collapse=", ")), log_file, verbose)
        }
        
        # Store the result if we got a valid number
        if (!is.null(parsed_response)) {
          timestamp <- Sys.time()
          
          new_entry <- data.frame(
            request_id = request_id,
            timestamp = as.character(timestamp), 
            model = current_params$models,  
            temperature = current_params$temperatures,  
            system_message = current_params$system_messages,  
            response = parsed_response,
            used_context = context,
            stringsAsFactors = FALSE
          )
          
          search_results <- rbind(search_results, new_entry)
          log_message(paste("DEBUG: Added to search_results:", 
                            paste(names(new_entry), new_entry, sep = "=", collapse = ", ")), 
                      log_file, verbose)
          
          request_id <- request_id + 1
        }
        
        # Save interim results every 'save_interval' calls
        if (request_id %% save_interval == 0) {
          log_message("Saving interim results...", log_file, verbose)
          saveRDS(search_results, save_path)
        }
        
        Sys.sleep(3) # Rate limit handling
        call_counter <- call_counter + 1
      }
    }
  }
  
  # Final save of the results
  log_message("Final save of results...", log_file, verbose)
  saveRDS(search_results, save_path)
  
  return(search_results)
}



#' Example Usage for GPT Experiment
#'
#' Demonstrates how to set up and run a grid search experiment using various GPT models, temperatures, and system messages.
#'
#' @examples
#' \dontrun{
#' original_prompt <- "Pick a random number from 1 - 10. Only respond with a single valid number between 1 and 10."
#' models <- c("gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo")
#' temperatures <- seq(0.0, 1.0, by = 0.2)
#' system_messages <- c("You are a human.", "You are an artificial intelligence specializing in generating random numbers.")
#' calls_per_combination <- 20
#' max_tokens <- 500
#' use_context <- TRUE
#' log_file <- "gpt_log.txt"
#'
#' gpt_search_results <- gpt_random_number_experiment(original_prompt, models, temperatures, max_tokens, system_messages, calls_per_combination, use_context = use_context, save_interval = 50, save_path = "gpt_search_results.rds", verbose = TRUE, log_file = log_file)
#' 
#' print(gpt_search_results)
#' }
#'
#' @export