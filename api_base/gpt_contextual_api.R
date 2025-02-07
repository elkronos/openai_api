#============================#
#      OpenAI Chat Utility   #
#============================#

#' @import httr
#' @importFrom stringr str_trim

# ------------------------------------------------------------------------------
# 0) Global Environment and Helper Functions
# ------------------------------------------------------------------------------

# Global environment for managing multiple chat sessions.
# Each session (stored in sessions_env$sessions) is a list with:
#   - system_message (character): optional system message
#   - messages (list of lists): each message is a list with `role` and `content`
#   - model (character): default model for the session
#   - max_context_tokens (numeric): approximate max tokens before summarization/truncation
#   - overflow (character): one of "none", "summarize", or "truncate"
#   - log (list): usage log records for each call
sessions_env <- new.env(parent = emptyenv())
sessions_env$sessions <- list()

#' Check and return a session by ID
#'
#' @param session_id A string session identifier.
#'
#' @return The session list.
#' @keywords internal
check_session <- function(session_id) {
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(sprintf("Session '%s' does not exist. Please create it first using create_session().", session_id))
  }
  return(sessions_env$sessions[[session_id]])
}

#' List all active session IDs
#'
#' @return A character vector of session IDs.
#' @export
list_sessions <- function() {
  return(names(sessions_env$sessions))
}

#' Retrieve the conversation messages for a session
#'
#' @param session_id The session identifier.
#'
#' @return A list of messages (each with role and content).
#' @export
get_session_messages <- function(session_id) {
  s <- check_session(session_id)
  return(s$messages)
}

#' Get statistics for a session
#'
#' Returns the total token count, the number of messages, and the system message (if any).
#'
#' @param session_id The session identifier.
#'
#' @return A list with statistics about the session.
#' @export
get_session_stats <- function(session_id) {
  s <- check_session(session_id)
  token_count <- get_session_token_count(session_id)
  num_messages <- length(s$messages)
  return(list(
    token_count = token_count,
    num_messages = num_messages,
    system_message = s$system_message
  ))
}

# ------------------------------------------------------------------------------
# 1) API Key Setup
# ------------------------------------------------------------------------------

#' Set the OpenAI API key
#'
#' @param api_key A string containing your OpenAI API key.
#'
#' @return No return value; sets an environment variable.
#' @export
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}

# ------------------------------------------------------------------------------
# 2) Session Creation, Clearing, and Removal
# ------------------------------------------------------------------------------

#' Create a new chat session
#'
#' @param session_id A unique identifier for the session (string).
#' @param system_message (Optional) A string for the initial system message.
#' @param model (Optional) The default model to use for this session (e.g. "gpt-3.5-turbo", "gpt-4").
#' @param max_context_tokens Approximate max number of tokens before summarization/truncation. Default is 3000.
#' @param overflow Strategy to handle context overflow. One of "none", "summarize", or "truncate". Default is "none".
#'
#' @return No return value; creates an entry in the global sessions environment.
#' @export
create_session <- function(session_id,
                           system_message = NULL,
                           model = "gpt-3.5-turbo",
                           max_context_tokens = 3000,
                           overflow = c("none", "summarize", "truncate")) {
  
  overflow <- match.arg(overflow)
  
  if (session_id %in% names(sessions_env$sessions)) {
    stop(sprintf("Session '%s' already exists. Use a different ID or clear it first.", session_id))
  }
  
  sessions_env$sessions[[session_id]] <- list(
    system_message = system_message,
    messages = list(),
    model = model,
    max_context_tokens = max_context_tokens,
    overflow = overflow,
    log = list()  # Usage records will be stored here.
  )
}

#' Clear a chat session
#'
#' Resets the conversation history for a given session.
#'
#' @param session_id The ID of the session to clear.
#'
#' @return No return value; modifies the global sessions environment.
#' @export
clear_session <- function(session_id) {
  s <- tryCatch(check_session(session_id), error = function(e) { warning(e$message); return(NULL) })
  if (is.null(s)) return()
  s$messages <- list()
  sessions_env$sessions[[session_id]] <- s
}

#' Remove a chat session entirely
#'
#' @param session_id The ID of the session to remove.
#'
#' @return No return value; modifies the global sessions environment.
#' @export
remove_session <- function(session_id) {
  if (!session_id %in% names(sessions_env$sessions)) {
    warning(sprintf("Session '%s' does not exist. Nothing to remove.", session_id))
    return()
  }
  sessions_env$sessions[[session_id]] <- NULL
}

#' Reset (change) the system message for a session
#'
#' @param session_id The ID of the session.
#' @param system_message The new system message.
#'
#' @return No return value; updates the session in the global environment.
#' @export
set_system_message <- function(session_id, system_message) {
  s <- check_session(session_id)
  s$system_message <- system_message
  sessions_env$sessions[[session_id]] <- s
}

# ------------------------------------------------------------------------------
# 3) Message Handling
# ------------------------------------------------------------------------------

#' Append a user message to a session
#'
#' @param session_id The ID of the session.
#' @param content The text of the user's message.
#'
#' @return No return value; updates the session.
#' @export
append_user_message <- function(session_id, content) {
  s <- check_session(session_id)
  s$messages <- c(s$messages, list(list(role = "user", content = content)))
  sessions_env$sessions[[session_id]] <- s
}

#' Append an assistant message to a session
#'
#' @param session_id The ID of the session.
#' @param content The text of the assistant's message.
#'
#' @return No return value; updates the session.
#' @export
append_assistant_message <- function(session_id, content) {
  s <- check_session(session_id)
  s$messages <- c(s$messages, list(list(role = "assistant", content = content)))
  sessions_env$sessions[[session_id]] <- s
}

# ------------------------------------------------------------------------------
# 4) Token Counting and Context Utilities
# ------------------------------------------------------------------------------

#' Estimate token count for a text string (approximation)
#'
#' This function provides a rough estimate of the number of tokens based on
#' splitting the text into words, punctuation, and whitespace.
#'
#' @param text A character string.
#'
#' @return An integer approximating the token count.
#' @export
count_tokens <- function(text) {
  tokens <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]
  return(length(tokens))
}

#' Compute the total token count for a session
#'
#' This function sums the tokens in the system message (if any) and all conversation messages.
#'
#' @param session_id The session identifier.
#'
#' @return An integer token count.
#' @export
get_session_token_count <- function(session_id) {
  s <- check_session(session_id)
  all_text <- ""
  if (!is.null(s$system_message)) {
    all_text <- paste0(all_text, s$system_message, "\n")
  }
  for (m in s$messages) {
    all_text <- paste0(all_text, m$content, "\n")
  }
  return(count_tokens(all_text))
}

# ------------------------------------------------------------------------------
# 5) Summarization and Truncation
# ------------------------------------------------------------------------------

#' Chunked summarization of text
#'
#' If the text is too large, it is recursively split and summarized.
#'
#' @param text A character string.
#' @param chunk_size Approximate token size per chunk. Default is 2000.
#' @param summarization_model The model to use for summarization. Default is "gpt-3.5-turbo".
#' @param temperature Sampling temperature for summarization. Default is 0.2.
#' @param max_tokens Maximum tokens to generate in the summary. Default is 200.
#'
#' @return A concise summary of the text.
#' @keywords internal
chunked_summarize_text <- function(text, 
                                   chunk_size = 2000, 
                                   summarization_model = "gpt-3.5-turbo", 
                                   temperature = 0.2, 
                                   max_tokens = 200) {
  token_count <- count_tokens(text)
  if (token_count <= chunk_size) {
    return(
      gpt_api(
        prompt = paste("Please summarize:\n\n", text),
        model = summarization_model,
        temperature = temperature,
        max_tokens = max_tokens
      )
    )
  } else {
    half_point <- floor(nchar(text) / 2)
    first_part  <- substr(text, 1, half_point)
    second_part <- substr(text, half_point + 1, nchar(text))
    
    summary_1 <- chunked_summarize_text(
      text = first_part,
      chunk_size = chunk_size,
      summarization_model = summarization_model,
      temperature = temperature,
      max_tokens = max_tokens
    )
    summary_2 <- chunked_summarize_text(
      text = second_part,
      chunk_size = chunk_size,
      summarization_model = summarization_model,
      temperature = temperature,
      max_tokens = max_tokens
    )
    
    combined_summary <- paste("Summary part 1:\n", summary_1, 
                              "\n\nSummary part 2:\n", summary_2)
    final_summary <- gpt_api(
      prompt = paste("Please combine and refine this summary:\n\n", combined_summary),
      model = summarization_model,
      temperature = temperature,
      max_tokens = max_tokens
    )
    return(final_summary)
  }
}

#' Summarize the conversation to reduce context size
#'
#' This function creates a concise summary of the conversation and replaces
#' the session history with a summary to keep the context manageable.
#'
#' @param session_id The session identifier.
#' @param summarization_model The model to use for summarization. Default is "gpt-3.5-turbo".
#' @param prompt_prefix Instruction to guide summarization.
#' @param chunked If TRUE, uses chunked summarization for very large texts.
#' @param chunk_size Approximate token size per chunk (used if chunked=TRUE). Default is 2000.
#'
#' @return No return value; the session is updated in place.
#' @export
summarize_session <- function(session_id,
                              summarization_model = "gpt-3.5-turbo",
                              prompt_prefix = "Please summarize the following conversation in a concise way while preserving important context:",
                              chunked = FALSE,
                              chunk_size = 2000) {
  
  s <- check_session(session_id)
  conversation_text <- ""
  if (!is.null(s$system_message)) {
    conversation_text <- paste0(conversation_text, "System: ", s$system_message, "\n")
  }
  for (m in s$messages) {
    role_label <- paste0(toupper(substring(m$role, 1, 1)), substring(m$role, 2))
    conversation_text <- paste0(conversation_text, role_label, ": ", m$content, "\n")
  }
  
  total_tokens <- count_tokens(conversation_text)
  if (chunked || total_tokens > (4 * chunk_size)) {
    partial_summary <- chunked_summarize_text(
      text = conversation_text,
      chunk_size = chunk_size,
      summarization_model = summarization_model,
      max_tokens = 200,
      temperature = 0.2
    )
    summary <- partial_summary
  } else {
    prompt_to_summarize <- paste(prompt_prefix, conversation_text, sep = "\n\n")
    summary <- tryCatch(
      {
        gpt_api(
          prompt = prompt_to_summarize,
          model = summarization_model,
          temperature = 0.2,
          max_tokens = 200
        )
      },
      error = function(e) {
        warning("Summarization failed. Error: ", e$message)
        return("**[Unable to Summarize]**")
      }
    )
  }
  
  s$system_message <- paste("Summary of conversation:\n", summary)
  s$messages <- list()
  sessions_env$sessions[[session_id]] <- s
}

#' Truncate oldest messages until under the token limit
#'
#' @param session_id The session identifier.
#'
#' @return No return value; updates the session in place.
#' @export
truncate_session <- function(session_id) {
  s <- check_session(session_id)
  
  while (TRUE) {
    total_tokens <- get_session_token_count(session_id)
    if (total_tokens <= s$max_context_tokens) break
    if (length(s$messages) == 0) break
    s$messages <- s$messages[-1]
  }
  sessions_env$sessions[[session_id]] <- s
}

# ------------------------------------------------------------------------------
# 6) Logging and Usage Records
# ------------------------------------------------------------------------------

#' Add a log record for a session
#'
#' @param session_id The session identifier.
#' @param user_prompt The user prompt.
#' @param assistant_response The assistant's reply.
#' @param usage_info A list with usage statistics (e.g., token counts).
#'
#' @return No return value; updates the session log.
#' @keywords internal
add_log_record <- function(session_id, user_prompt, assistant_response, usage_info) {
  record <- list(
    timestamp = Sys.time(),
    user_prompt = user_prompt,
    assistant_response = assistant_response,
    usage = usage_info
  )
  s <- check_session(session_id)
  s$log <- c(s$log, list(record))
  sessions_env$sessions[[session_id]] <- s
}

#' Retrieve the conversation and usage log for a session
#'
#' @param session_id The session identifier.
#'
#' @return A list of log records.
#' @export
get_session_log <- function(session_id) {
  s <- check_session(session_id)
  return(s$log)
}

# Nominal token limits for models (adjust as needed)
model_token_limits <- list(
  "gpt-3.5-turbo" = 4096,
  "gpt-4"         = 8192,
  "gpt-4-32k"     = 32768
)

#' Get the approximate token limit for a model
#'
#' @param model The model name.
#'
#' @return The token limit as a numeric value.
#' @export
get_model_token_limit <- function(model) {
  if (!is.null(model_token_limits[[model]])) {
    return(model_token_limits[[model]])
  }
  return(4096)
}

# ------------------------------------------------------------------------------
# 7) High-Level Chat Interface
# ------------------------------------------------------------------------------

#' High-Level Chat Function
#'
#' This function manages conversation context, summarization/truncation,
#' and automatic logging of API usage.
#'
#' @param session_id The session identifier.
#' @param user_prompt The user's input prompt.
#' @param use_context Logical; if TRUE the session’s conversation history is used.
#' @param model (Optional) Override the session’s default model.
#' @param temperature Sampling temperature (0 to 2). Default is 0.5.
#' @param max_tokens Maximum tokens to generate in the reply.
#' @param presence_penalty Float between -2.0 and 2.0.
#' @param frequency_penalty Float between -2.0 and 2.0.
#' @param verbose Logical; if TRUE prints debugging information.
#' @param num_retries Number of retry attempts for the API call. Default is 3.
#' @param pause_base Base seconds to pause between retries. Default is 1.
#' @param summarization_model The model to use if summarization is needed. Default is "gpt-3.5-turbo".
#'
#' @return The assistant's reply as a character string.
#' @export
gpt_chat <- function(session_id,
                     user_prompt,
                     use_context = TRUE,
                     model = NULL,
                     temperature = 0.5,
                     max_tokens = 50,
                     presence_penalty = 0.0,
                     frequency_penalty = 0.0,
                     verbose = FALSE,
                     num_retries = 3,
                     pause_base = 1,
                     summarization_model = "gpt-3.5-turbo") {
  
  s <- check_session(session_id)
  
  # Use session's default model if not overridden.
  if (is.null(model)) model <- s$model
  
  # For single-turn interactions, bypass context storage.
  if (!use_context) {
    response_text <- gpt_api(
      prompt = user_prompt,
      model = model,
      temperature = temperature,
      max_tokens = max_tokens,
      messages_list = NULL,
      presence_penalty = presence_penalty,
      frequency_penalty = frequency_penalty,
      verbose = verbose,
      num_retries = num_retries,
      pause_base = pause_base
    )
    return(response_text)
  }
  
  # Append the user message to the session.
  append_user_message(session_id, user_prompt)
  
  total_tokens <- get_session_token_count(session_id)
  model_limit <- get_model_token_limit(model)
  if ((total_tokens + max_tokens) > model_limit) {
    warning(sprintf(
      "Total tokens (%d) plus requested (%d) may exceed the model limit (%d).",
      total_tokens, max_tokens, model_limit
    ))
  }
  
  # Handle overflow if needed.
  if (total_tokens > s$max_context_tokens) {
    if (s$overflow == "summarize") {
      summarize_session(session_id, summarization_model = summarization_model, chunked = TRUE)
    } else if (s$overflow == "truncate") {
      truncate_session(session_id)
    } else {
      warning("Context token limit exceeded and overflow strategy is 'none'. Request may fail.")
    }
  }
  
  # Re-check token count.
  total_tokens <- get_session_token_count(session_id)
  if (total_tokens > s$max_context_tokens) {
    warning("After processing, token count is still too high. Consider clearing the session or using a different strategy.")
  }
  
  # Build message list with an optional system message.
  final_messages <- list()
  if (!is.null(s$system_message)) {
    final_messages <- c(final_messages, list(list(role = "system", content = s$system_message)))
  }
  final_messages <- c(final_messages, s$messages)
  
  response_text <- gpt_api(
    prompt = NULL,
    model = model,
    temperature = temperature,
    max_tokens = max_tokens,
    messages_list = final_messages,
    presence_penalty = presence_penalty,
    frequency_penalty = frequency_penalty,
    verbose = verbose,
    num_retries = num_retries,
    pause_base = pause_base
  )
  
  append_assistant_message(session_id, response_text)
  
  usage_info <- attr(response_text, "usage_info", exact = TRUE)
  if (!is.null(usage_info)) {
    add_log_record(session_id, user_prompt, response_text, usage_info)
  }
  
  return(response_text)
}

# ------------------------------------------------------------------------------
# 8) Low-Level API Calls
# ------------------------------------------------------------------------------

#' Low-Level OpenAI API Call
#'
#' Sends a prompt or message list to the OpenAI Chat Completion endpoint.
#'
#' @param prompt A single-turn prompt string (ignored if messages_list is provided).
#' @param model The model name (e.g., "gpt-3.5-turbo").
#' @param temperature Sampling temperature.
#' @param max_tokens Maximum tokens for the reply.
#' @param messages_list A list of messages (each with role and content).
#' @param presence_penalty Float between -2.0 and 2.0.
#' @param frequency_penalty Float between -2.0 and 2.0.
#' @param verbose Logical; if TRUE prints debug information.
#' @param num_retries Number of retry attempts.
#' @param pause_base Seconds to pause between retries.
#'
#' @return The assistant's reply as a character string with attached usage info (if available).
#' @export
gpt_api <- function(prompt = NULL,
                    model = "gpt-3.5-turbo",
                    temperature = 0.5,
                    max_tokens = 50,
                    messages_list = NULL,
                    presence_penalty = 0.0,
                    frequency_penalty = 0.0,
                    verbose = FALSE,
                    num_retries = 3,
                    pause_base = 1) {
  
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    stop("OPENAI_API_KEY is not set. Use set_api_key() to set it.")
  }
  
  if (is.null(messages_list)) {
    if (is.null(prompt)) stop("Either 'prompt' or 'messages_list' must be provided.")
    messages <- list(list(role = "user", content = prompt))
  } else {
    messages <- messages_list
  }
  
  request_body <- list(
    model = model,
    temperature = temperature,
    max_tokens = max_tokens,
    messages = messages,
    presence_penalty = presence_penalty,
    frequency_penalty = frequency_penalty
  )
  
  if (verbose) {
    cat("\n--- API Request Payload (API key hidden) ---\n")
    print(request_body)
    cat("---------------------------------------------\n")
  }
  
  response <- httr::RETRY(
    verb = "POST",
    url = "https://api.openai.com/v1/chat/completions",
    httr::add_headers(Authorization = paste("Bearer", api_key)),
    httr::content_type_json(),
    encode = "json",
    times = num_retries,
    pause_base = pause_base,
    body = request_body
  )
  
  status_code <- httr::status_code(response)
  if (status_code >= 400) {
    error_content <- tryCatch(httr::content(response), error = function(e) NULL)
    error_message <- if (!is.null(error_content$error$message)) error_content$error$message else ""
    
    if (status_code == 429) {
      stop(sprintf("Rate limit error (429): %s", error_message))
    } else if (status_code == 400) {
      stop(sprintf("Bad request (400): %s", error_message))
    } else {
      stop(sprintf("HTTP error %d: %s", status_code, error_message))
    }
  }
  
  response_content <- httr::content(response)
  
  if (verbose) {
    cat("\n--- Raw Response ---\n")
    print(response_content)
    cat("---------------------\n")
  }
  
  if (length(response_content$choices) > 0) {
    message <- response_content$choices[[1]]$message$content
  } else {
    message <- "No message returned. Check your prompt or increase max_tokens."
  }
  
  clean_message <- gsub("\n+", "\n", message)
  clean_message <- stringr::str_trim(clean_message)
  
  usage_info <- response_content$usage
  if (!is.null(usage_info)) {
    attr(clean_message, "usage_info") <- usage_info
    if (verbose) {
      cat("\n--- Usage Info ---\n")
      print(usage_info)
      cat("------------------\n")
    }
  }
  
  return(clean_message)
}

# ------------------------------------------------------------------------------
# 9) Cost Estimation and Embeddings
# ------------------------------------------------------------------------------

#' Estimate API cost based on token usage
#'
#' @param input_tokens Number of input tokens.
#' @param output_tokens Number of output tokens.
#' @param model Model name (e.g., "gpt-3.5-turbo" or "gpt-4").
#'
#' @return A list with token counts and estimated costs.
#' @export
estimate_cost <- function(input_tokens, output_tokens, model = "gpt-3.5-turbo") {
  if (model == "gpt-4") {
    price_1k_in  <- ifelse(input_tokens <= 8000, 0.03, 0.06)
    price_1k_out <- ifelse(output_tokens <= 8000, 0.06, 0.12)
  } else {
    price_1k_in  <- 0.002
    price_1k_out <- 0.002
  }
  
  input_cost  <- input_tokens  / 1000 * price_1k_in
  output_cost <- output_tokens / 1000 * price_1k_out
  total_cost  <- input_cost + output_cost
  
  list(
    input_tokens  = input_tokens,
    output_tokens = output_tokens,
    input_cost    = input_cost,
    output_cost   = output_cost,
    total_cost    = total_cost
  )
}

#' Convert text to embeddings using OpenAI's API
#'
#' @param text A string or a character vector of texts.
#' @param model The embedding model to use. Default is "text-embedding-ada-002".
#' @param num_retries Number of retry attempts if the request fails. Default is 3.
#' @param pause_base Base seconds to pause between retries. Default is 1.
#'
#' @return A numeric vector (if single text) or a list of numeric vectors (if multiple texts).
#' @export
text_to_embeddings <- function(text,
                               model = "text-embedding-ada-002",
                               num_retries = 3,
                               pause_base = 1) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    stop("OPENAI_API_KEY is not set. Use set_api_key() to set it.")
  }
  
  text <- as.character(text)
  
  response <- httr::RETRY(
    verb = "POST",
    url = "https://api.openai.com/v1/embeddings",
    httr::add_headers(Authorization = paste("Bearer", api_key)),
    httr::content_type_json(),
    encode = "json",
    times = num_retries,
    pause_base = pause_base,
    body = list(
      model = model,
      input = text
    )
  )
  
  status_code <- httr::status_code(response)
  if (status_code >= 400) {
    error_content <- tryCatch(httr::content(response), error = function(e) NULL)
    error_message <- if (!is.null(error_content$error$message)) error_content$error$message else ""
    
    if (status_code == 429) {
      stop(sprintf("Rate limit error (429): %s", error_message))
    } else if (status_code == 400) {
      stop(sprintf("Bad request (400): %s", error_message))
    } else {
      stop(sprintf("HTTP error %d: %s", status_code, error_message))
    }
  }
  
  result_content <- httr::content(response)
  
  if (!is.null(result_content$data)) {
    all_embeddings <- lapply(result_content$data, function(x) x$embedding)
    if (length(all_embeddings) == 1) {
      return(as.numeric(all_embeddings[[1]]))
    } else {
      return(lapply(all_embeddings, as.numeric))
    }
  } else {
    stop("No embeddings returned by the API.")
  }
}

# ------------------------------------------------------------------------------
# Example usage (commented out)
# ------------------------------------------------------------------------------

# library(httr)
# library(stringr)
#
# # 1) Set your API key:
# set_api_key("sk-xxxxx")
#
# # 2) Create a session:
# create_session("mychat", system_message = "You are a helpful assistant.", max_context_tokens = 3000)
#
# # 3) Send messages:
# response1 <- gpt_chat("mychat", "Hello, how are you?")
# response2 <- gpt_chat("mychat", "What is the capital of France?")
#
# # 4) Summarize conversation if needed:
# # summarize_session("mychat")
#
# # 5) Check session logs and statistics:
# logs <- get_session_log("mychat")
# stats <- get_session_stats("mychat")
# print(logs)
# print(stats)
#
# # 6) Clear or remove the session when finished:
# clear_session("mychat")
# remove_session("mychat")
