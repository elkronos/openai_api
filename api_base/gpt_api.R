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

#' @title Count Tokens
#' 
#' @description
#' Provides a rough estimate of the number of tokens in a text string by splitting
#' on words, punctuation, and whitespace. 
#' 
#' @details
#' This is only an approximation and may not match exactly how the OpenAI tokenizer
#' counts tokens. 
#' 
#' @importFrom base strsplit
#' 
#' @param text A character string to estimate the token count for.
#' 
#' @return An integer representing the approximate token count.
#' 
#' @examples
#' \dontrun{
#' n <- count_tokens("Hello, world!")
#' print(n)
#' }
count_tokens <- function(text) {
  tokens <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]
  length(tokens)
}

#' @title Estimate Cost
#' 
#' @description
#' Calculates approximate cost given a model, input token count, and output token count.
#' 
#' @details
#' Prices are approximate and based on known OpenAI rates for GPT-3.5-turbo and GPT-4.
#' 
#' @param input_tokens Integer. Number of input tokens.
#' @param output_tokens Integer. Number of output tokens.
#' @param model A character string specifying the model (e.g., `"gpt-3.5-turbo"`, `"gpt-4"`).
#' 
#' @return A list with elements `input_tokens`, `output_tokens`, `input_cost`,
#'   `output_cost`, `total_cost`.
#' 
#' @examples
#' \dontrun{
#' est <- estimate_cost(1200, 800, "gpt-4")
#' print(est)
#' }
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
  list(
    input_tokens  = input_tokens,
    output_tokens = output_tokens,
    input_cost    = input_cost,
    output_cost   = output_cost,
    total_cost    = input_cost + output_cost
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
