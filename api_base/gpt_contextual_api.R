#============================#
#      OpenAI Chat Utility   #
#============================#

#' @import httr
#' @importFrom stringr str_trim

# ============================
#  1) Set the OpenAI API Key
# ============================
#' Set the OpenAI API key
#'
#' @param api_key A string containing your OpenAI API key.
#'
#' @return No return value, sets an environment variable.
#' @export
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}

# Global environment for managing multiple chat sessions
# We store all sessions in `sessions_env$sessions`.
# Each session is a list with:
#   - system_message (character): optional system role content
#   - messages (list of lists): conversation messages, each with `role` and `content`
#   - model (character): default model
#   - max_context_tokens (numeric): approximate max tokens allowed before summarizing/truncating
#   - overflow (character): "none", "summarize", or "truncate"
#   - log (list): optional usage log records for each call
sessions_env <- new.env(parent = emptyenv())
sessions_env$sessions <- list()

# ======================================
# 2) Create, Clear, Remove Chat Sessions
# ======================================

#' Create a new chat session
#'
#' @param session_id A unique identifier for the session (string).
#' @param system_message (Optional) A string for the initial system message.
#' @param model (Optional) The default model to use for this session (e.g. "gpt-3.5-turbo", "gpt-4").
#' @param max_context_tokens Approximate max number of tokens for the session before summarization/truncation. Default 3000.
#' @param overflow Strategy to handle overflow. One of "none", "summarize", "truncate". Default "none".
#'
#' @return No return value, but creates an entry in the global sessions environment.
#' @export
create_session <- function(session_id,
                           system_message = NULL,
                           model = "gpt-3.5-turbo",
                           max_context_tokens = 3000,
                           overflow = c("none", "summarize", "truncate")) {
  
  overflow <- match.arg(overflow)
  
  if (session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "already exists. Use a different ID or clear it first."))
  }
  
  sessions_env$sessions[[session_id]] <- list(
    system_message = system_message,
    messages = list(),
    model = model,
    max_context_tokens = max_context_tokens,
    overflow = overflow,
    log = list()  # We'll store usage records etc. here
  )
}

#' Clear a chat session
#'
#' This resets the conversation history for a given session.
#'
#' @param session_id The ID of the session to clear.
#'
#' @return No return value, modifies the global sessions environment.
#' @export
clear_session <- function(session_id) {
  if (!session_id %in% names(sessions_env$sessions)) {
    warning(paste("Session", session_id, "does not exist. Nothing to clear."))
    return()
  }
  
  sessions_env$sessions[[session_id]]$messages <- list()
}

#' Remove a chat session entirely
#'
#' @param session_id The ID of the session to remove.
#'
#' @return No return value, modifies the global sessions environment.
#' @export
remove_session <- function(session_id) {
  if (!session_id %in% names(sessions_env$sessions)) {
    warning(paste("Session", session_id, "does not exist. Nothing to remove."))
    return()
  }
  
  sessions_env$sessions[[session_id]] <- NULL
}

#' Reset (change) the system message for a session
#'
#' @param session_id The ID of the session.
#' @param system_message The new system message to set.
#'
#' @return No return value; updates the session in the global environment.
#' @export
set_system_message <- function(session_id, system_message) {
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist."))
  }
  sessions_env$sessions[[session_id]]$system_message <- system_message
}


# ==================================
# 3) Utility to Append Messages
# ==================================

#' Utility function to append a user message
#'
#' @param session_id The ID of the session to which the message will be appended.
#' @param content The text of the user's message.
#'
#' @return No return value, updates the session in the global environment.
#' @export
append_user_message <- function(session_id, content) {
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist."))
  }
  
  sessions_env$sessions[[session_id]]$messages <- c(
    sessions_env$sessions[[session_id]]$messages,
    list(list(role = "user", content = content))
  )
}

#' Utility function to append an assistant message
#'
#' @param session_id The ID of the session to which the message will be appended.
#' @param content The text of the assistant's message.
#'
#' @return No return value, updates the session in the global environment.
#' @export
append_assistant_message <- function(session_id, content) {
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist."))
  }
  
  sessions_env$sessions[[session_id]]$messages <- c(
    sessions_env$sessions[[session_id]]$messages,
    list(list(role = "assistant", content = content))
  )
}


# =====================================
# 4) Approximate Token Counting & Tools
# =====================================

#' Estimate token count of a text string (approximation)
#'
#' This function provides a rough estimate of the number of tokens in a text string, 
#' based on splitting the text into words, punctuation and whitespaces. 
#' Note that this is an approximation and may not match the token count used 
#' by specific language models.
#'
#' @param text A character string for which to estimate the token count.
#'
#' @return An integer giving the estimated number of tokens in the input text.
#' @export
count_tokens <- function(text) {
  # Split text into words, punctuation, and whitespaces
  tokens <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]
  return(length(tokens))
}

# Helper: Summation of token counts for a session
get_session_token_count <- function(session_id) {
  s <- sessions_env$sessions[[session_id]]
  if (is.null(s)) {
    return(0)
  }
  
  # Combine all message contents plus optional system message
  all_text <- ""
  if (!is.null(s$system_message)) {
    all_text <- paste0(all_text, s$system_message, "\n")
  }
  for (m in s$messages) {
    all_text <- paste0(all_text, m$content, "\n")
  }
  
  return(count_tokens(all_text))
}


# ===================================
# 5) Summaries & Chunked Summaries
# ===================================

# Helper for chunked summarization: split a large text into manageable tokens
# Summarize each chunk, then optionally do a final summary of the partial summaries.
chunked_summarize_text <- function(text, 
                                   chunk_size = 2000, 
                                   summarization_model = "gpt-3.5-turbo", 
                                   temperature = 0.2, 
                                   max_tokens = 200) {
  # Split text by lines or sentences, then recombine into ~chunk_size token blocks
  # For simplicity, we'll do a naive approach: if count_tokens > chunk_size, break it in half, etc.
  
  token_count <- count_tokens(text)
  if (token_count <= chunk_size) {
    # We can summarize the entire thing at once
    return(
      gpt_api(
        prompt = paste("Please summarize:\n\n", text),
        model = summarization_model,
        temperature = temperature,
        max_tokens = max_tokens
      )
    )
  } else {
    # Split in half and summarize each half
    half_point <- floor(nchar(text) / 2)
    # This is a very naive approach: we split by character length, not tokens. 
    # A more robust approach would chunk by actual tokens. But let's keep it simple for demonstration.
    
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
    
    # Now summarize the two summaries
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


#' Summarize the conversation in a session to reduce token usage
#'
#' This function takes the existing conversation and creates a concise summary
#' to preserve the context while reducing token count. It replaces the full
#' conversation history with a single system message containing that summary.
#'
#' @param session_id The session ID.
#' @param summarization_model Which model to use for the summarization step. Default "gpt-3.5-turbo".
#' @param prompt_prefix A string to instruct the model on how to summarize.
#' @param chunked If TRUE, performs chunked summarization to handle very large text.
#' @param chunk_size Approx token size per chunk (used if chunked=TRUE). Default 2000.
#'
#' @return No return value, modifies the session in place.
#' @export
summarize_session <- function(session_id,
                              summarization_model = "gpt-3.5-turbo",
                              prompt_prefix = "Please summarize the following conversation in a concise way while preserving important context:",
                              chunked = FALSE,
                              chunk_size = 2000) {
  
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist."))
  }
  
  s <- sessions_env$sessions[[session_id]]
  
  # Combine all messages into a single string
  conversation_text <- ""
  if (!is.null(s$system_message)) {
    conversation_text <- paste0(conversation_text, "System: ", s$system_message, "\n")
  }
  for (m in s$messages) {
    conversation_text <- paste0(
      conversation_text,
      toupper(substring(m$role, 1, 1)), substring(m$role, 2),
      ": ", m$content, "\n"
    )
  }
  
  total_tokens <- count_tokens(conversation_text)
  # If it's obviously huge, handle chunked summarization
  if (chunked || total_tokens > (4 * chunk_size)) {
    # Summarize in chunks
    partial_summary <- chunked_summarize_text(
      text = conversation_text,
      chunk_size = chunk_size,
      summarization_model = summarization_model,
      max_tokens = 200,
      temperature = 0.2
    )
    summary <- partial_summary
  } else {
    # Summarize in one go
    prompt_to_summarize <- paste(prompt_prefix, conversation_text, sep = "\n\n")
    # The summarization could fail if it's too large, so we might catch errors:
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
        warning("Summarization call failed. Consider chunked summarization or manual intervention. Error:\n", e$message)
        return("**[Unable to Summarize - Too Large or Error Occurred]**")
      }
    )
  }
  
  # Replace the entire conversation with one system message = summary
  s$system_message <- paste("Summary of conversation:\n", summary)
  s$messages <- list()
  
  sessions_env$sessions[[session_id]] <- s
}


#' Truncate the oldest messages in a session until under the token limit
#'
#' @param session_id The session ID.
#'
#' @return No return value, modifies the session in place.
truncate_session <- function(session_id) {
  s <- sessions_env$sessions[[session_id]]
  
  # Keep removing from the front of messages until under the limit
  while (TRUE) {
    total_tokens <- get_session_token_count(session_id)
    if (total_tokens <= s$max_context_tokens) {
      break
    }
    if (length(s$messages) == 0) {
      # If there's nothing left to remove, we can't do more
      break
    }
    # Remove the oldest message (front of the list)
    s$messages <- s$messages[-1]
  }
  
  sessions_env$sessions[[session_id]] <- s
}


# ================================================
# 6) Manage Chat Flow with Automatic Logging
# ================================================

# Helper: store a log record of usage, tokens, etc.
add_log_record <- function(session_id, user_prompt, assistant_response, usage_info) {
  # usage_info might be something like: list(prompt_tokens=37, completion_tokens=29, total_tokens=66)
  record <- list(
    timestamp = Sys.time(),
    user_prompt = user_prompt,
    assistant_response = assistant_response,
    usage = usage_info
  )
  sessions_env$sessions[[session_id]]$log <- c(
    sessions_env$sessions[[session_id]]$log,
    list(record)
  )
}

#' Retrieve the usage and conversation log for a session
#'
#' @param session_id The ID of the session.
#'
#' @return A list of log records, each with timestamp, user_prompt, assistant_response, usage, etc.
#' @export
get_session_log <- function(session_id) {
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist."))
  }
  return(sessions_env$sessions[[session_id]]$log)
}

# Some nominal token limits (for warning checks)
# Adjust as needed for your environment / new model versions
model_token_limits <- list(
  "gpt-3.5-turbo" = 4096,
  "gpt-4"         = 8192,
  "gpt-4-32k"     = 32768
)

get_model_token_limit <- function(model) {
  if (!is.null(model_token_limits[[model]])) {
    return(model_token_limits[[model]])
  }
  # fallback if unknown model
  return(4096)
}

#' Higher-level Chat Function: Manages conversation context, summarization, and truncation
#'
#' @param session_id The ID of the session to use.
#' @param user_prompt The user's prompt or message for the assistant.
#' @param use_context Logical. If TRUE, will use the session's messages as context. If FALSE, sends only the current user prompt.
#' @param model (Optional) Override the session's default model for this request.
#' @param temperature Sampling temperature (0 to 2). Defaults to 0.5.
#' @param max_tokens Maximum tokens to generate in the response.
#' @param presence_penalty Float between -2.0 and 2.0
#' @param frequency_penalty Float between -2.0 and 2.0
#' @param verbose If TRUE, prints debugging information (request payload, raw response).
#' @param num_retries Number of times to retry the request in case of failure. Default is 3.
#' @param pause_base Seconds to wait between retries. Default is 1.
#' @param summarization_model The model to use if summarizing the conversation is needed. Default "gpt-3.5-turbo".
#'
#' @return The assistant's response as a character string.
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
  
  if (!session_id %in% names(sessions_env$sessions)) {
    stop(paste("Session", session_id, "does not exist. Please create_session() first."))
  }
  
  s <- sessions_env$sessions[[session_id]]
  
  # If model is not specified, use the session's default
  if (is.null(model)) {
    model <- s$model
  }
  
  # If use_context = FALSE, we skip storing conversation
  if (!use_context) {
    # Just do a single-turn prompt
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
    # Optionally log usage in a minimal way, but there's no "context" per se
    # We'll do it if we get usage from gpt_api return. See below changes in gpt_api.
    return(response_text)
  }
  
  # ========== If using context ==========
  append_user_message(session_id, user_prompt)
  
  total_tokens <- get_session_token_count(session_id)
  # Additional guard: check model limit (approx)
  model_limit <- get_model_token_limit(model)
  if ((total_tokens + max_tokens) > model_limit) {
    warning(sprintf(
      "Requested tokens (%d) plus conversation size (%d) likely exceeds the model limit (%d). Consider summarizing or truncating further.",
      max_tokens, total_tokens, model_limit
    ))
  }
  
  # If we exceed session's max_context_tokens, do summarization or truncation
  if (total_tokens > s$max_context_tokens) {
    if (s$overflow == "summarize") {
      summarize_session(session_id, summarization_model = summarization_model, chunked = TRUE)
    } else if (s$overflow == "truncate") {
      truncate_session(session_id)
    } else {
      # s$overflow == "none"
      warning("Context token limit exceeded, but overflow strategy is 'none'. The request may fail.")
    }
  }
  
  # Re-check after summarization/truncation
  total_tokens <- get_session_token_count(session_id)
  if (total_tokens > s$max_context_tokens) {
    warning("Even after attempting to shorten context, token count is still too high. Consider clearing the session or using a different strategy.")
  }
  
  # Build final message list (system + conversation)
  final_messages <- list()
  if (!is.null(s$system_message)) {
    final_messages <- c(final_messages, list(list(role = "system", content = s$system_message)))
  }
  final_messages <- c(final_messages, s$messages)
  
  # Make the API call with context
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
  
  # Store the assistant's response in the session
  append_assistant_message(session_id, response_text)
  
  # If usage_info was returned, store a log record
  usage_info <- attr(response_text, "usage_info", exact = TRUE)
  if (!is.null(usage_info)) {
    add_log_record(
      session_id = session_id,
      user_prompt = user_prompt,
      assistant_response = response_text,
      usage_info = usage_info
    )
  }
  
  return(response_text)
}


# ======================================
# 7) Low-Level gpt_api Function
# ======================================

#' Low-level function: Send prompt/messages to the OpenAI Chat Completion endpoint
#'
#' This is the core function that performs the API call. For multi-turn context,
#' use \code{messages_list}. For a single prompt, use \code{prompt}.
#'
#' @param prompt A string with the user's text (if you just want a single-turn prompt).
#' @param model Model name, e.g. "gpt-3.5-turbo" or "gpt-4".
#' @param temperature Sampling temperature (0 to 2).
#' @param max_tokens Max tokens to generate in the response.
#' @param messages_list A list of messages, each with \code{role} and \code{content}. If provided, \code{prompt} is ignored.
#' @param presence_penalty Float between -2.0 and 2.0
#' @param frequency_penalty Float between -2.0 and 2.0
#' @param verbose If TRUE, prints debug info (payload minus the API key, and raw response).
#' @param num_retries Number of times to retry on error (429, timeouts, etc.).
#' @param pause_base Seconds to wait between retries.
#'
#' @return The content of the assistant's reply. 
#'         If usage info is returned by the API, that is attached as an attribute "usage_info".
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
  
  # Require the API key to be set
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    stop("OPENAI_API_KEY is not set. Please set it with set_api_key().")
  }
  
  # Build the 'messages' payload:
  # If messages_list is not NULL, use it directly.
  # Otherwise, create a single user message from 'prompt'.
  if (is.null(messages_list)) {
    if (is.null(prompt)) {
      stop("Either 'prompt' or 'messages_list' must be provided.")
    }
    messages <- list(list(role = "user", content = prompt))
  } else {
    messages <- messages_list
  }
  
  # Construct request body
  request_body <- list(
    model = model,
    temperature = temperature,
    max_tokens = max_tokens,
    messages = messages,
    presence_penalty = presence_penalty,
    frequency_penalty = frequency_penalty
  )
  
  # Verbose logging: show the payload minus the API key
  if (verbose) {
    cat("\n--- API Request Payload ---\n")
    print(request_body)
    cat("---------------------------\n")
  }
  
  # Perform the POST with retries
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
  
  # Check the status code for errors
  status_code <- httr::status_code(response)
  if (status_code >= 400) {
    # Attempt to parse error content
    error_content <- tryCatch(httr::content(response), error = function(e) NULL)
    error_message <- ""
    if (!is.null(error_content$error$message)) {
      error_message <- error_content$error$message
    }
    
    if (status_code == 429) {
      stop(paste("Rate limit error (429):", error_message))
    } else if (status_code == 400) {
      # Possibly the prompt/context is too large, or some bad request
      stop(paste("Bad request (400):", error_message))
    } else {
      stop(paste("HTTP error", status_code, "-", error_message))
    }
  }
  
  # Parse the response content
  response_content <- httr::content(response)
  
  if (verbose) {
    cat("\n--- Raw Response ---\n")
    print(response_content)
    cat("---------------------\n")
  }
  
  # Extract the assistant message
  if (length(response_content$choices) > 0) {
    message <- response_content$choices[[1]]$message$content
  } else {
    message <- "The model did not return a message. Try increasing max_tokens or check your prompt."
  }
  
  # Attach usage info if present
  usage_info <- response_content$usage  # might be NULL if not returned
  # usage_info could be something like: list(prompt_tokens=..., completion_tokens=..., total_tokens=...)
  
  # Clean up the message
  clean_message <- gsub("\n+", "\n", message)  # keep line breaks but compress multiples
  clean_message <- stringr::str_trim(clean_message)
  
  # We can attach usage info as an attribute
  if (!is.null(usage_info)) {
    attr(clean_message, "usage_info") <- usage_info
    if (verbose) {
      cat("\n--- Usage ---\n")
      print(usage_info)
      cat("----------------\n")
    }
  }
  
  return(clean_message)
}


# =======================================
# 8) Cost Estimation & Embeddings
# =======================================

#' Estimate the cost based on input and output tokens for given model
#'
#' @param input_tokens The number of input tokens.
#' @param output_tokens The number of output tokens.
#' @param model Model name, e.g., "gpt-3.5-turbo" or "gpt-4".
#'
#' @return A list with fields for \code{input_tokens}, \code{output_tokens}, 
#'         \code{input_cost}, \code{output_cost}, and \code{total_cost}.
#' @export
estimate_cost <- function(input_tokens, output_tokens, model = "gpt-3.5-turbo") {
  
  # Rough cost structure as of early 2023
  if (model == "gpt-4") {
    # Different rates if context is beyond 8k tokens
    price_1k_in  <- ifelse(input_tokens <= 8000, 0.03, 0.06)
    price_1k_out <- ifelse(output_tokens <= 8000, 0.06, 0.12)
  } else {
    # GPT-3.5-turbo rates
    price_1k_in  <- 0.002
    price_1k_out <- 0.002
  }
  
  input_cost  <- input_tokens  / 1000 * price_1k_in
  output_cost <- output_tokens / 1000 * price_1k_out
  total_cost  <- input_cost + output_cost
  
  result <- list(
    input_tokens   = input_tokens,
    output_tokens  = output_tokens,
    input_cost     = input_cost,
    output_cost    = output_cost,
    total_cost     = total_cost
  )
  
  return(result)
}

#' Convert text to embeddings using OpenAI's API
#'
#' @param text A string or character vector of text to embed.
#' @param model The embedding model to use. Default "text-embedding-ada-002".
#' @param num_retries Number of retry attempts if the request fails. Default 3.
#' @param pause_base Base pause between retries in seconds. Default 1.
#'
#' @return A numeric vector (if a single text) or a list of numeric vectors (if multiple texts).
#' @export
text_to_embeddings <- function(text,
                               model = "text-embedding-ada-002",
                               num_retries = 3,
                               pause_base = 1) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") {
    stop("OPENAI_API_KEY is not set. Please set it with set_api_key().")
  }
  
  # If 'text' is a single string, coerce to a length-1 character vector
  if (length(text) == 1) {
    text <- as.character(text)
  }
  
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
  
  # Check for HTTP errors
  status_code <- httr::status_code(response)
  if (status_code >= 400) {
    error_content <- tryCatch(httr::content(response), error = function(e) NULL)
    error_message <- ""
    if (!is.null(error_content$error$message)) {
      error_message <- error_content$error$message
    }
    if (status_code == 429) {
      stop(paste("Rate limit error (429):", error_message))
    } else if (status_code == 400) {
      stop(paste("Bad request (400):", error_message))
    } else {
      stop(paste("HTTP error", status_code, "-", error_message))
    }
  }
  
  result_content <- httr::content(response)
  
  # Each entry in data has an embedding field
  if (!is.null(result_content$data)) {
    all_embeddings <- lapply(result_content$data, function(x) x$embedding)
    # If we only asked for a single text, return a numeric vector
    if (length(all_embeddings) == 1) {
      return(as.numeric(all_embeddings[[1]]))
    } else {
      # Otherwise return a list of numeric vectors
      return(lapply(all_embeddings, as.numeric))
    }
  } else {
    stop("The API did not return any embeddings.")
  }
}


# ==============================
# Example usage (commented out)
# ==============================
# library(httr)
# library(stringr)
#
# # 1) Set API key (replace "sk-xxxxx" with your actual key)
# set_api_key("sk-xxxxx")
#
# # 2) Create a session
# create_session("mychat", system_message = "You are a helpful assistant.", max_context_tokens = 3000)
#
# # 3) Chat with gpt_chat
# response1 <- gpt_chat("mychat", "Hello, how are you?")
# response2 <- gpt_chat("mychat", "What is the capital of France?")
#
# # 4) Summarize if the conversation becomes large
# # summarize_session("mychat")
#
# # 5) Check logs
# logs <- get_session_log("mychat")
# print(logs)
#
# # 6) Clear or Remove session if needed
# clear_session("mychat")
# remove_session("mychat")
