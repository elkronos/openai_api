#' @title Set OpenAI API Key
#' 
#' @description
#' Stores the OpenAI API key in an environment variable (`OPENAI_API_KEY`).
#' 
#' @details
#' This function sets the `OPENAI_API_KEY` environment variable, which is used by
#' other functions in this script to authenticate with the OpenAI API.
#' 
#' @importFrom base Sys.setenv
#' 
#' @param api_key A character string containing your OpenAI API key.
#' 
#' @return
#' Invisibly returns `NULL` (called for side-effects).
#' 
#' @examples
#' \dontrun{
#' set_api_key("sk-xxxxx")
#' }
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}

#' @title List Available Models
#' 
#' @description
#' Retrieves the list of models accessible to your OpenAI account.
#' 
#' @details
#' This function sends a GET request to the `/v1/models` endpoint and parses
#' the returned JSON into a data frame. Only models your account can query
#' are typically returned, but some models may still be inaccessible.
#' 
#' @importFrom httr RETRY stop_for_status content add_headers
#' @importFrom stats setNames
#' @importFrom base paste
#' @importFrom jsonlite fromJSON
#' 
#' @return A data frame with columns such as `id`, `object`, `created`, and `owned_by`.
#' 
#' @examples
#' \dontrun{
#' models_df <- list_available_models()
#' head(models_df)
#' }
list_available_models <- function() {
  r <- RETRY(
    "GET",
    url = "https://api.openai.com/v1/models",
    add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY"))),
    encode    = "json",
    times     = 3,
    pause_base= 1
  )
  stop_for_status(r)
  d <- content(r, as = "parsed", encoding = "UTF-8")
  if (is.null(d$data)) stop("No model data returned.")
  df <- do.call(rbind, lapply(d$data, as.data.frame, stringsAsFactors = FALSE))
  rownames(df) <- NULL
  df
}

#' @title Filter and Sort Models
#' 
#' @description
#' Filters out undesirable models (e.g. preview, audio, dalle, etc.) and sorts
#' the remaining models by a custom priority order, placing non-mini versions
#' above "mini" variants, then sorting by creation time descending.
#' 
#' @details
#' This function allows you to remove models based on substrings in their `id`,
#' exclude those owned by `openai-internal`, and group them by patterns in
#' `custom_order`. Models that do not match any custom pattern are put at the
#' bottom of the list. Within the same group, non-mini is preferred over mini,
#' and more recently created models appear first.
#' 
#' @importFrom dplyr arrange %>%
#' @importFrom stringr str_detect
#' @importFrom base grepl
#' 
#' @param df A data frame of models, typically from `list_available_models()`.
#' @param custom_order A character vector specifying group patterns in priority order.
#'   Defaults to `c("3o","1o","4","3")`.
#' @param skip_preview Logical. If TRUE, drops models whose IDs contain `"preview"`.
#' @param skip_audio Logical. If TRUE, drops models whose IDs contain `"audio"`.
#' @param skip_dalle Logical. If TRUE, drops models whose IDs contain `"dalle"`.
#' @param skip_whisper Logical. If TRUE, drops models whose IDs contain `"whisper"`.
#' @param skip_babbage Logical. If TRUE, drops models whose IDs contain `"babbage"`.
#' @param skip_tts Logical. If TRUE, drops models whose IDs contain `"tts"`.
#' @param skip_moderation Logical. If TRUE, drops models whose IDs contain `"moderation"`.
#' @param skip_internal Logical. If TRUE, excludes rows where `owned_by == "openai-internal"`.
#' 
#' @return A data frame sorted by group order, non-mini vs mini, and descending `created`.
#' 
#' @examples
#' \dontrun{
#' all_models <- list_available_models()
#' sorted_models <- filter_and_sort_models(all_models)
#' head(sorted_models)
#' }
filter_and_sort_models <- function(
    df,
    custom_order    = c("3o","1o","4","3"),
    skip_preview    = TRUE,
    skip_audio      = TRUE,
    skip_dalle      = TRUE,
    skip_whisper    = TRUE,
    skip_babbage    = TRUE,
    skip_tts        = TRUE,
    skip_moderation = TRUE,
    skip_internal   = TRUE
) {
  forbidden_patterns <- c()
  if (skip_preview)    forbidden_patterns <- c(forbidden_patterns, "preview")
  if (skip_audio)      forbidden_patterns <- c(forbidden_patterns, "audio")
  if (skip_dalle)      forbidden_patterns <- c(forbidden_patterns, "dalle")
  if (skip_whisper)    forbidden_patterns <- c(forbidden_patterns, "whisper")
  if (skip_babbage)    forbidden_patterns <- c(forbidden_patterns, "babbage")
  if (skip_tts)        forbidden_patterns <- c(forbidden_patterns, "tts")
  if (skip_moderation) forbidden_patterns <- c(forbidden_patterns, "moderation")
  
  if (length(forbidden_patterns) > 0) {
    rx <- paste(forbidden_patterns, collapse = "|")
    df <- subset(df, !grepl(rx, id, ignore.case = TRUE))
  }
  if (skip_internal) {
    df <- subset(df, !(owned_by %in% "openai-internal"))
  }
  if (nrow(df) == 0) stop("No models left after filtering.")
  
  get_group_index <- function(x) {
    for (i in seq_along(custom_order)) {
      if (grepl(custom_order[i], x, ignore.case = TRUE)) return(i)
    }
    length(custom_order) + 1
  }
  
  df$group_index <- sapply(df$id, get_group_index)
  df$is_mini     <- ifelse(grepl("mini", df$id, ignore.case = TRUE), 1, 0)
  
  df %>% arrange(
    group_index,
    is_mini,
    desc(created)
  )
}

#' @title Create a Chat Completion
#' 
#' @description
#' Sends a chat request to the OpenAI `/v1/chat/completions` endpoint using the
#' specified model and parameters.
#' 
#' @details
#' This function uses the updated `max_completion_tokens` parameter (rather than
#' the deprecated `max_tokens`). It can handle transient errors using `httr::RETRY`.
#' 
#' @importFrom httr RETRY stop_for_status content add_headers content_type_json
#' @importFrom base paste
#' 
#' @param messages A list of role/content pairs (system/user/assistant). 
#' @param model A string specifying which model to use (e.g. `"gpt-3.5-turbo"`).
#' @param temperature A numeric controlling randomness. Default is 0.7.
#' @param max_completion_tokens An integer limiting generated tokens. Default is 50.
#' 
#' @return A parsed list from the JSON response. Typically includes fields like
#'   `choices`, `usage`, `model`, etc.
#' 
#' @examples
#' \dontrun{
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user",   content = "Hello!")
#' )
#' resp <- create_chat_completion(msgs, "gpt-3.5-turbo", temperature=0.6, max_completion_tokens=100)
#' }
create_chat_completion <- function(
    messages,
    model,
    temperature           = 0.7,
    max_completion_tokens = 1024
) {
  r <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY"))),
    content_type_json(),
    encode = "json",
    times = 3,
    pause_base = 1,
    body = list(
      model                 = model,
      messages              = messages,
      temperature           = temperature,
      max_completion_tokens = max_completion_tokens
    )
  )
  stop_for_status(r)
  content(r, as = "parsed", encoding = "UTF-8")
}

#' @title Extract Assistant Message
#' 
#' @description
#' Pulls the assistant's text message from a chat completion response.
#' 
#' @details
#' If the response includes at least one choice with a non-null `message$content`,
#' this function returns that text. Otherwise returns `NULL`.
#' 
#' @param resp A parsed object from `create_chat_completion()` or 
#'   `create_chat_completion_with_failover()`.
#' 
#' @return A character string with the assistant's generated text, or `NULL` if none.
#' 
#' @examples
#' \dontrun{
#' assistant_text <- extract_assistant_message(resp)
#' cat(assistant_text)
#' }
extract_assistant_message <- function(resp) {
  if (length(resp$choices) > 0 && !is.null(resp$choices[[1]]$message$content)) {
    resp$choices[[1]]$message$content
  } else {
    NULL
  }
}

#' @title Create Chat Completion with Failover
#' 
#' @description
#' Tries a user-specified model first, if any, then optionally falls back to an
#' automatically filtered and sorted list of models until one succeeds.
#' 
#' @details
#' 
#' 1. If `use_best_model=FALSE`, it calls `create_chat_completion()` directly on
#'    `model`. If `model` is not supplied, it errors out.
#'
#' 2. If `use_best_model=TRUE` and a specific `model` is provided, it attempts
#'    that model first. On error, it proceeds to filter/sort all available models
#'    and tries each in turn.
#'
#' 3. If `use_best_model=TRUE` and `model=NULL`, it skips directly to filtering and
#'    sorting all models. 
#'
#' The filtering excludes certain model IDs (e.g. containing "preview", "audio",
#' "dalle", etc.) and optionally excludes "openai-internal". Sorting is controlled
#' by patterns in `custom_order`, grouping them, then ranking non-mini over mini,
#' and sorting by newest first within each group.
#' 
#' @param messages A list of role/content pairs (system/user/assistant).
#' @param model An optional string. If provided and `use_best_model=TRUE`, it tries
#'   this model first, then falls back. If `use_best_model=FALSE`, it only uses 
#'   this model.
#' @param use_best_model Logical. If TRUE, triggers the failover logic.
#' @param custom_order A character vector specifying which patterns define the
#'   model priority grouping, in order. Defaults to `c("3o","1o","4","3")`.
#' @param skip_preview Logical. If TRUE, remove IDs containing "preview".
#' @param skip_audio Logical. If TRUE, remove IDs containing "audio".
#' @param skip_dalle Logical. If TRUE, remove IDs containing "dalle".
#' @param skip_whisper Logical. If TRUE, remove IDs containing "whisper".
#' @param skip_babbage Logical. If TRUE, remove IDs containing "babbage".
#' @param skip_tts Logical. If TRUE, remove IDs containing "tts".
#' @param skip_moderation Logical. If TRUE, remove IDs containing "moderation".
#' @param skip_internal Logical. If TRUE, remove rows with `owned_by == "openai-internal"`.
#' @param temperature Numeric. Passed to `create_chat_completion()`. Defaults to 0.7.
#' @param max_completion_tokens Integer. Passed to `create_chat_completion()`. Defaults to 50.
#' 
#' @return The parsed JSON response from the first successful model. If all fail,
#'   it raises an error.
#' 
#' @examples
#' \dontrun{
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user",   content = "Hello!")
#' )
#'
#' # Auto-failover
#' res <- create_chat_completion_with_failover(msgs, use_best_model=TRUE)
#'
#' # Force a specific model, with fallback
#' res2 <- create_chat_completion_with_failover(msgs, model="gpt-4", use_best_model=TRUE)
#'
#' # Force only a specific model, no fallback
#' res3 <- create_chat_completion_with_failover(msgs, model="gpt-3.5-turbo", use_best_model=FALSE)
#' }
create_chat_completion_with_failover <- function(
    messages,
    model                 = NULL,
    use_best_model        = TRUE,
    custom_order          = c("3o","1o","4","3"),
    skip_preview          = TRUE,
    skip_audio            = TRUE,
    skip_dalle            = TRUE,
    skip_whisper          = TRUE,
    skip_babbage          = TRUE,
    skip_tts              = TRUE,
    skip_moderation       = TRUE,
    skip_internal         = TRUE,
    temperature           = 0.7,
    max_completion_tokens = 1024
) {
  if (!use_best_model) {
    if (is.null(model)) stop("No model specified, and use_best_model=FALSE.")
    return(create_chat_completion(messages, model, temperature, max_completion_tokens))
  } else if (!is.null(model)) {
    result <- tryCatch(
      create_chat_completion(messages, model, temperature, max_completion_tokens),
      error = function(e) e
    )
    if (!inherits(result, "error")) {
      return(result)
    }
  }
  d  <- list_available_models()
  df <- filter_and_sort_models(
    d, custom_order,
    skip_preview, skip_audio, skip_dalle, skip_whisper, skip_babbage, skip_tts, skip_moderation, skip_internal
  )
  for (i in seq_len(nrow(df))) {
    cid <- df$id[i]
    out <- tryCatch(
      create_chat_completion(messages, cid, temperature, max_completion_tokens),
      error = function(e) e
    )
    if (!inherits(out, "error")) {
      return(out)
    }
  }
  stop("All candidate models failed.")
}

#' @title Enhanced Token Estimation
#'
#' @description
#' Approximates how OpenAI's GPT models tokenize text by splitting on word
#' boundaries, punctuation, common special characters, and repeated letters.
#'
#' @details
#' This is still only an approximation of how GPT-based Byte-Pair Encoding (BPE)
#' or TikToken might split text. For the most accurate count, you would need to
#' call OpenAI's tokenizer API directly (which may not be available in all
#' languages).
#'
#' The function:
#' \itemize{
#'   \item Breaks on most punctuation and whitespace
#'   \item Captures repeated sequences of letters/numbers
#'   \item Attempts to handle special or repeated patterns that might be
#'         sub-tokenized (e.g., repeated exclamation marks, repeated letters)
#' }
#'
#' @param text A character string to estimate token count for.
#' @param include_legacy Logical. If TRUE, also runs a simpler fallback split
#'   for comparison and returns both counts.
#'
#' @return If \code{include_legacy=FALSE}, returns a single integer. If
#'   \code{include_legacy=TRUE}, returns a named integer vector with elements
#'   \code{enhanced_count} and \code{legacy_count}.
#'
#' @examples
#' \dontrun{
#' txt <- "Hello!!! This is a test 123 123!"
#' token_count <- enhanced_token_count(txt, include_legacy = TRUE)
#' print(token_count)
#' }
enhanced_token_count <- function(text, include_legacy = FALSE) {
  # Basic approach tries to capture repeated punctuation / special chars / digits.
  #
  # The big idea: break on sequences of letters vs. digits vs. punctuation,
  # and also split repeated punctuation, etc.
  #
  # One example of a rough BPE-like pattern:
  #   - A sequence of letters: [A-Za-z]+
  #   - A sequence of digits: [0-9]+
  #   - Punctuation, possibly repeated: [[:punct:]]+
  #
  # We'll stitch them into one big pattern, then remove empty tokens.
  pattern <- "([[:alpha:]]+)|([[:digit:]]+)|([[:punct:]]+)|([[:space:]]+)"
  
  # Extract all matches. Each match is a separate token chunk.
  # We'll exclude purely whitespace matches from the final count.
  matches <- gregexpr(pattern, text, perl = TRUE)
  pieces <- regmatches(text, matches)[[1]]
  
  # Remove pure spaces
  pieces <- pieces[!grepl("^[[:space:]]+$", pieces)]
  
  # For repeated punctuation, we might want to further split each chunk:
  # e.g., "!!!" => "!", "!", "!"
  # We'll do that with another pass if punctuation chunk is longer than 1 char.
  final_tokens <- lapply(pieces, function(x) {
    if (grepl("^[[:punct:]]+$", x) && nchar(x) > 1) {
      # separate each punctuation char
      unlist(strsplit(x, "", fixed = TRUE))
    } else {
      x
    }
  })
  final_tokens <- unlist(final_tokens)
  
  # For legacy comparison:
  if (include_legacy) {
    # The simpler old version:
    legacy_split <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]
    # Clean out empties
    legacy_split <- legacy_split[nzchar(legacy_split)]
    c(
      enhanced_count = length(final_tokens),
      legacy_count   = length(legacy_split)
    )
  } else {
    length(final_tokens)
  }
}

#' @title GPT API Wrapper
#'
#' @description
#' Provides a single function interface (`gpt_api()`) that can:
#' - Set the API key
#' - List available models
#' - Filter/sort models
#' - Create a chat completion
#' - Extract the assistant's text from a response
#' - Create a chat completion with failover
#' - Estimate token counts
#' - Estimate cost
#' - Generate embeddings
#'
#' @details
#' Call `gpt_api()` with the `action` argument specifying which functionality you want.
#' All other arguments can be passed via `...` and are forwarded to the underlying function.
#'
#' Supported `action` values:
#' \itemize{
#'   \item \strong{"set_api_key"} — sets the API key (calls \code{set_api_key}).
#'   \item \strong{"list_models"} — retrieves model list (calls \code{list_available_models}).
#'   \item \strong{"filter_models"} — filters/sorts a models data frame (calls \code{filter_and_sort_models}).
#'   \item \strong{"create_chat"} — sends a chat request (calls \code{create_chat_completion}).
#'   \item \strong{"extract_message"} — extracts assistant text (calls \code{extract_assistant_message}).
#'   \item \strong{"create_chat_failover"} — chat with model failover (calls \code{create_chat_completion_with_failover}).
#'   \item \strong{"estimate_tokens"} — estimates token count (calls \code{enhanced_token_count}).
#'   \item \strong{"estimate_cost"} — estimates cost (calls \code{estimate_cost}).
#'   \item \strong{"embedding"} — generates embeddings (calls \code{text_to_embeddings}).
#' }
#'
#' @param action A character string specifying which helper function to call.
#' @param ... Additional arguments passed on to the underlying helper function.
#'
#' @return Varies depending on `action`.
#'
#' @examples
#' \dontrun{
#' # Set API key
#' gpt_api("set_api_key", api_key = "sk-xxxxx")
#'
#' # List all models
#' mods <- gpt_api("list_models")
#' head(mods)
#'
#' # Create a chat
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user",   content = "Hello!")
#' )
#' resp <- gpt_api("create_chat", messages = msgs, model = "gpt-3.5-turbo")
#' cat(gpt_api("extract_message", resp = resp))
#'
#' # Estimate cost
#' cost_info <- gpt_api("estimate_cost", input_tokens=1200, output_tokens=800, model="gpt-3.5-turbo")
#' print(cost_info)
#' }
gpt_api <- function(action, ...) {
  switch(
    EXPR = action,
    
    "set_api_key" = {
      set_api_key(...)
    },
    
    "list_models" = {
      list_available_models(...)
    },
    
    "filter_models" = {
      filter_and_sort_models(...)
    },
    
    "create_chat" = {
      create_chat_completion(...)
    },
    
    "extract_message" = {
      extract_assistant_message(...)
    },
    
    "create_chat_failover" = {
      create_chat_completion_with_failover(...)
    },
    
    "estimate_tokens" = {
      enhanced_token_count(...)
    },
    
    "estimate_cost" = {
      estimate_cost(...)
    },
    
    "embedding" = {
      text_to_embeddings(...)
    },
    
    {
      stop(
        "Unknown action: '", action,
        "'. Valid actions are: set_api_key, list_models, filter_models, ",
        "create_chat, extract_message, create_chat_failover, estimate_tokens, ",
        "estimate_cost, embedding."
      )
    }
  )
}


#' @title Estimate Cost
#' 
#' @description
#' Calculates approximate cost based on a model's per-token rates and the specified
#' number of input and output tokens. This version includes updated rates for newer
#' “gpt-4o” and “gpt-4.5-preview” style models, as well as others you listed.
#' 
#' @details
#' Prices are expressed per 1 million tokens in your table. This function converts
#' them to per 1k tokens internally. It attempts to match the provided model name
#' via partial pattern matching. If no match is found, it falls back to the original
#' GPT-3.5 and GPT-4 logic. You can add or remove entries as your needs change.
#'
#' @importFrom base grepl
#' 
#' @param input_tokens  Integer. Number of input tokens.
#' @param output_tokens Integer. Number of output tokens.
#' @param model         Character string specifying the model name/ID.
#' 
#' @return A list with:
#' \describe{
#'   \item{input_tokens}{Integer, the number of input tokens.}
#'   \item{output_tokens}{Integer, the number of output tokens.}
#'   \item{input_cost}{Numeric, the approximate cost for input tokens.}
#'   \item{output_cost}{Numeric, the approximate cost for output tokens.}
#'   \item{total_cost}{Numeric, sum of input_cost + output_cost.}
#' }
#' 
#' @examples
#' \dontrun{
#' estimate_cost(1200, 800, "gpt-4o")
#' estimate_cost(1200, 800, "gpt-4.5-preview")
#' estimate_cost(1200, 800, "gpt-3.5-turbo")
#' }
estimate_cost <- function(input_tokens, output_tokens, model = "gpt-3.5-turbo") {
  
  # Table of partial patterns vs. cost per 1k tokens (input, output)
  # Model: Price per 1M => dividing by 1000 => per 1k tokens
  # Adjust or extend as needed
  cost_map <- list(
    list(pattern = "gpt-4.5-preview",     input = 0.075,  output = 0.15),
    list(pattern = "gpt-4o-mini",         input = 0.00015, output = 0.0006),
    list(pattern = "gpt-4o-realtime",     input = 0.005,   output = 0.02),
    list(pattern = "gpt-4o",              input = 0.0025,  output = 0.01),
    list(pattern = "o1-mini",             input = 0.0011,  output = 0.0044),
    list(pattern = "o3-mini",             input = 0.0011,  output = 0.0044),
    list(pattern = "o1",                  input = 0.015,   output = 0.06),
    list(pattern = "gpt-3.5-turbo",       input = 0.0005,  output = 0.0015),
    list(pattern = "gpt-4",               input = 0.03,    output = 0.06)
  )
  
  # Defaults if no pattern is matched
  price_1k_in  <- 0.002  # older “GPT-3” style or fallback
  price_1k_out <- 0.002
  
  # Attempt partial match
  lower_model <- tolower(model)
  for (entry in cost_map) {
    if (grepl(entry$pattern, lower_model, fixed = FALSE)) {
      price_1k_in  <- entry$input
      price_1k_out <- entry$output
      break
    }
  }
  
  # Compute cost
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



#' @title Convert Text to Embeddings
#' 
#' @description
#' Sends text to the OpenAI `/v1/embeddings` endpoint and retrieves a numerical embedding vector.
#' 
#' @details
#' The function uses `httr::RETRY` to handle transient API failures. 
#' It only returns the first embedding if multiple are requested.
#' 
#' @importFrom httr RETRY stop_for_status content add_headers content_type_json
#' @importFrom base as.numeric
#' 
#' @param text A character string to embed. 
#' @param model A character specifying which embedding model to use. Defaults to `"text-embedding-ada-002"`.
#' @param num_retries An integer. Number of retries for transient errors. Defaults to 3.
#' @param pause_base A numeric. Base seconds between retries. Defaults to 1.
#' 
#' @return A numeric vector of embeddings.
#' 
#' @examples
#' \dontrun{
#' emb <- text_to_embeddings("Hello world")
#' print(length(emb))
#' }
text_to_embeddings <- function(text, model = "text-embedding-ada-002", num_retries = 3, pause_base = 1) {
  r <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/embeddings",
    add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY"))),
    content_type_json(),
    encode = "json",
    times = num_retries,
    pause_base = pause_base,
    body = list(
      model = model,
      input = text
    )
  )
  stop_for_status(r)
  parsed <- content(r, as = "parsed", encoding = "UTF-8")
  if (length(parsed$data) == 0) stop("No embeddings returned.")
  as.numeric(parsed$data[[1]]$embedding)
}

#' @examples
#' \dontrun{
#' # Example usage:
#' # messages <- list(
#' #   list(role = "system", content = "You are a helpful assistant."),
#' #   list(role = "user",   content = "Write a short limerick about data science.")
#' # )
#' # out <- create_chat_completion_with_failover(messages, use_best_model = TRUE, max_completion_tokens = 100)
#' # cat(extract_assistant_message(out))
#' #
#' # cost_est <- estimate_cost(1200, 800, "gpt-3.5-turbo")
#' # print(cost_est)
#' #
#' # emb <- text_to_embeddings("Hello world")
#' # str(emb)
#' }
NULL
