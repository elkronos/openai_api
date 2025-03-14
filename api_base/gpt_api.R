#' @title Set OpenAI API Key
#'
#' @description
#' Stores the OpenAI API key in an environment variable (`OPENAI_API_KEY`).
#'
#' @param api_key A character string containing your OpenAI API key.
#'
#' @return Invisibly returns `NULL`.
#'
#' @examples
#' \dontrun{
#' set_api_key("sk-xxxxx")
#' }
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}


#' @title List Models
#'
#' @description
#' Retrieves a data frame of available models from the OpenAI API.
#'
#' @return A data frame with columns such as `id`, `object`, `created`, and `owned_by`.
#'
#' @examples
#' \dontrun{
#' mods <- list_models()
#' head(mods)
#' }
list_models <- function() {
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


#' @title Filter Models
#'
#' @description
#' Filters out undesirable models (e.g., preview, audio, dalle, etc.) and sorts
#' the remaining models by a custom priority order.
#'
#' @param df A data frame of models from `list_models()`.
#' @param custom_order A character vector for priority grouping. Defaults to `c("3o", "1o", "4", "3")`.
#' @param skip_preview Logical; drop models with "preview" in the ID. Default TRUE.
#' @param skip_audio Logical; drop models with "audio" in the ID. Default TRUE.
#' @param skip_dalle Logical; drop models with "dalle" in the ID. Default TRUE.
#' @param skip_whisper Logical; drop models with "whisper" in the ID. Default TRUE.
#' @param skip_babbage Logical; drop models with "babbage" in the ID. Default TRUE.
#' @param skip_tts Logical; drop models with "tts" in the ID. Default TRUE.
#' @param skip_moderation Logical; drop models with "moderation" in the ID. Default TRUE.
#' @param skip_internal Logical; exclude models owned by "openai-internal". Default TRUE.
#'
#' @return A sorted data frame of models.
#'
#' @examples
#' \dontrun{
#' mods <- list_models()
#' filtered <- filter_models(mods)
#' head(filtered)
#' }
filter_models <- function(
    df,
    custom_order    = c("3o", "1o", "4", "3"),
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


#' @title Get Best Model
#'
#' @description
#' Returns the top-ranked model ID from the filtered and sorted list.
#'
#' @param custom_order A character vector for priority grouping. Defaults to `c("3o", "1o", "4", "3")`.
#' @param ... Additional arguments passed to `filter_models()`.
#'
#' @return A character string containing the best (latest) model ID.
#'
#' @examples
#' \dontrun{
#' best <- get_best_model()
#' print(best)
#' }
get_best_model <- function(custom_order = c("3o", "1o", "4", "3"), ...) {
  d <- list_models()
  df <- filter_models(d, custom_order = custom_order, ...)
  df$id[1]
}


#' @title Get Model Maximum Tokens
#'
#' @description
#' Returns the maximum allowed tokens for a given model.
#'
#' @param model A character string specifying the model name.
#'
#' @return An integer representing the maximum token limit for the model.
#'
#' @examples
#' \dontrun{
#' max_tokens <- get_model_max_tokens("gpt-3.5-turbo")
#' print(max_tokens)
#' }
get_model_max_tokens <- function(model) {
  # Define model-specific token limits (adjust as needed)
  limits <- list(
    "gpt-3.5-turbo"   = 4096,
    "gpt-4"           = 8192,
    "gpt-4.5-preview" = 8192
  )
  default_max <- 2048
  # Use lower case to compare
  limits[[tolower(model)]] %||% default_max
}


#' @title Chat
#'
#' @description
#' Sends a chat request to the OpenAI `/v1/chat/completions` endpoint.
#' If no model is provided, the best available model is chosen automatically.
#' If `max_completion_tokens` is not specified, a model-specific maximum is used.
#'
#' @param messages A list of role/content pairs.
#' @param model A character string specifying the model to use. Defaults to `NULL`.
#' @param temperature A numeric value controlling randomness. Default is 0.7.
#' @param max_completion_tokens An integer for the maximum tokens to generate.
#'   Defaults to the model-specific maximum if not provided.
#'
#' @return A parsed list from the JSON response.
#'
#' @examples
#' \dontrun{
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "Hello!")
#' )
#' resp <- chat(msgs)
#' }
chat <- function(
    messages,
    model                 = NULL,
    temperature           = 0.7,
    max_completion_tokens = NULL
) {
  if (is.null(model)) {
    model <- get_best_model()
  }
  if (is.null(max_completion_tokens)) {
    max_completion_tokens <- get_model_max_tokens(model)
  }
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


#' @title Extract
#'
#' @description
#' Extracts the assistant's text message from a chat response.
#'
#' @param resp A parsed response from `chat()` or `chat_failover()`.
#'
#' @return A character string containing the assistant's message, or `NULL` if not found.
#'
#' @examples
#' \dontrun{
#' txt <- extract(resp)
#' cat(txt)
#' }
extract <- function(resp) {
  if (length(resp$choices) > 0 && !is.null(resp$choices[[1]]$message$content)) {
    resp$choices[[1]]$message$content
  } else {
    NULL
  }
}


#' @title Chat Failover
#'
#' @description
#' Attempts to create a chat completion using a specified model first.
#' If that fails (or no model is provided), it iterates through available models until one succeeds.
#'
#' @param messages A list of role/content pairs.
#' @param model An optional character string for the initial model. Defaults to `NULL`.
#' @param use_best_model Logical. If TRUE, enables automatic failover. Default is TRUE.
#' @param custom_order A character vector for priority grouping. Defaults to `c("3o", "1o", "4", "3")`.
#' @param skip_preview, skip_audio, skip_dalle, skip_whisper, skip_babbage, skip_tts,
#'        skip_moderation, skip_internal Logical flags for filtering models. All default to TRUE.
#' @param temperature A numeric value for randomness. Default is 0.7.
#' @param max_completion_tokens An integer for maximum tokens to generate.
#'   Defaults to the model-specific maximum if not provided.
#'
#' @return A parsed response from the first successful chat call.
#'
#' @examples
#' \dontrun{
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "Hello!")
#' )
#' res <- chat_failover(msgs)
#' }
chat_failover <- function(
    messages,
    model                 = NULL,
    use_best_model        = TRUE,
    custom_order          = c("3o", "1o", "4", "3"),
    skip_preview          = TRUE,
    skip_audio            = TRUE,
    skip_dalle            = TRUE,
    skip_whisper          = TRUE,
    skip_babbage          = TRUE,
    skip_tts              = TRUE,
    skip_moderation       = TRUE,
    skip_internal         = TRUE,
    temperature           = 0.7,
    max_completion_tokens = NULL
) {
  if (!use_best_model) {
    if (is.null(model)) stop("No model specified and use_best_model=FALSE.")
    return(chat(messages, model, temperature, max_completion_tokens))
  } else if (!is.null(model)) {
    result <- tryCatch(
      chat(messages, model, temperature, max_completion_tokens),
      error = function(e) e
    )
    if (!inherits(result, "error")) return(result)
  }
  d  <- list_models()
  df <- filter_models(
    d, custom_order,
    skip_preview, skip_audio, skip_dalle, skip_whisper,
    skip_babbage, skip_tts, skip_moderation, skip_internal
  )
  for (i in seq_len(nrow(df))) {
    cid <- df$id[i]
    out <- tryCatch(
      chat(messages, cid, temperature, max_completion_tokens),
      error = function(e) e
    )
    if (!inherits(out, "error")) return(out)
  }
  stop("All candidate models failed.")
}


#' @title Token Count
#'
#' @description
#' Approximates token count by splitting text on word boundaries, punctuation, etc.
#' Optionally returns both an enhanced and a legacy count.
#'
#' @param text A character string for tokenization.
#' @param include_legacy Logical. If TRUE, returns both enhanced and legacy counts. Default is FALSE.
#'
#' @return Either an integer (enhanced count) or a named vector with `enhanced_count` and `legacy_count`.
#'
#' @examples
#' \dontrun{
#' token_count("Hello!!! This is a test.", include_legacy = TRUE)
#' }
token_count <- function(text, include_legacy = FALSE) {
  pattern <- "([[:alpha:]]+)|([[:digit:]]+)|([[:punct:]]+)|([[:space:]]+)"
  matches <- gregexpr(pattern, text, perl = TRUE)
  pieces <- regmatches(text, matches)[[1]]
  
  pieces <- pieces[!grepl("^[[:space:]]+$", pieces)]
  
  final_tokens <- unlist(lapply(pieces, function(x) {
    if (grepl("^[[:punct:]]+$", x) && nchar(x) > 1) {
      unlist(strsplit(x, "", fixed = TRUE))
    } else {
      x
    }
  }))
  
  if (include_legacy) {
    legacy_split <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]
    legacy_split <- legacy_split[nzchar(legacy_split)]
    c(
      enhanced_count = length(final_tokens),
      legacy_count   = length(legacy_split)
    )
  } else {
    length(final_tokens)
  }
}


#' @title Estimate Cost
#'
#' @description
#' Calculates approximate cost based on input/output tokens and model-specific pricing.
#'
#' @param input_tokens An integer number of input tokens.
#' @param output_tokens An integer number of output tokens.
#' @param model A character string specifying the model. Default is `"gpt-3.5-turbo"`.
#'
#' @return A list with input_tokens, output_tokens, input_cost, output_cost, and total_cost.
#'
#' @examples
#' \dontrun{
#' estimate_cost(1200, 800, "gpt-3.5-turbo")
#' }
estimate_cost <- function(input_tokens, output_tokens, model = "gpt-3.5-turbo") {
  cost_map <- list(
    list(pattern = "gpt-4.5-preview",     input = 0.075,   output = 0.15),
    list(pattern = "gpt-4o-mini",         input = 0.00015, output = 0.0006),
    list(pattern = "gpt-4o-realtime",     input = 0.005,   output = 0.02),
    list(pattern = "gpt-4o",              input = 0.0025,  output = 0.01),
    list(pattern = "o1-mini",             input = 0.0011,  output = 0.0044),
    list(pattern = "o3-mini",             input = 0.0011,  output = 0.0044),
    list(pattern = "o1",                  input = 0.015,   output = 0.06),
    list(pattern = "gpt-3.5-turbo",       input = 0.0005,  output = 0.0015),
    list(pattern = "gpt-4",               input = 0.03,    output = 0.06)
  )
  
  price_1k_in  <- 0.002
  price_1k_out <- 0.002
  
  lower_model <- tolower(model)
  for (entry in cost_map) {
    if (grepl(entry$pattern, lower_model)) {
      price_1k_in  <- entry$input
      price_1k_out <- entry$output
      break
    }
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


#' @title Embeddings
#'
#' @description
#' Sends text to the OpenAI `/v1/embeddings` endpoint and returns the embedding vector.
#'
#' @param text A character string to embed.
#' @param model A character specifying the embedding model. Default is `"text-embedding-ada-002"`.
#' @param num_retries Integer number of retries on failure. Default is 3.
#' @param pause_base Numeric seconds between retries. Default is 1.
#'
#' @return A numeric vector of embeddings.
#'
#' @examples
#' \dontrun{
#' emb <- embeddings("Hello world")
#' print(length(emb))
#' }
embeddings <- function(text, model = "text-embedding-ada-002", num_retries = 3, pause_base = 1) {
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


#' @title GPT API Wrapper
#'
#' @description
#' A unified interface for several OpenAI GPT functions. Supported actions include:
#'   - "set_api_key"
#'   - "list_models"
#'   - "filter_models"
#'   - "chat"
#'   - "extract"
#'   - "chat_failover"
#'   - "token_count"
#'   - "estimate_cost"
#'   - "embedding"
#'   - "help" (lists all available options)
#'
#' @param action A character string specifying the helper function to call.
#' @param ... Additional arguments for the chosen action.
#'
#' @return The output from the corresponding helper function.
#'
#' @examples
#' \dontrun{
#' # See available actions:
#' gpt_api("help")
#'
#' # Set API key:
#' gpt_api("set_api_key", api_key = "sk-xxxxx")
#'
#' # List models:
#' mods <- gpt_api("list_models")
#'
#' # Create a chat:
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "Hello!")
#' )
#' resp <- gpt_api("chat", messages = msgs)
#' cat(gpt_api("extract", resp = resp))
#'
#' # Estimate cost:
#' cost_info <- gpt_api("estimate_cost", input_tokens = 1200, output_tokens = 800, model = "gpt-3.5-turbo")
#'
#' # Get embeddings:
#' emb <- gpt_api("embedding", text = "Hello world")
#' }
gpt_api <- function(action, ...) {
  valid_actions <- c("set_api_key", "list_models", "filter_models", "chat",
                     "extract", "chat_failover", "token_count", "estimate_cost", "embedding", "help")
  
  if (missing(action) || action == "help") {
    message("Valid actions: ", paste(valid_actions, collapse = ", "))
    return(invisible(valid_actions))
  }
  
  switch(
    EXPR = action,
    
    "set_api_key"    = set_api_key(...),
    "list_models"    = list_models(...),
    "filter_models"  = filter_models(...),
    "chat"           = chat(...),
    "extract"        = extract(...),
    "chat_failover"  = chat_failover(...),
    "token_count"    = token_count(...),
    "estimate_cost"  = estimate_cost(...),
    "embedding"      = embeddings(...),
    
    stop("Unknown action: '", action,
         "'. Use gpt_api('help') to see valid actions.")
  )
}


#' @examples
#' \dontrun{
#' # Example usage:
#' # Set your API key:
#' gpt_api("set_api_key", api_key = "sk-xxxxx")
#'
#' # List available models:
#' mods <- gpt_api("list_models")
#' head(mods)
#'
#' # Create a chat:
#' msgs <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "Tell me a joke.")
#' )
#' resp <- gpt_api("chat", messages = msgs)
#' cat(gpt_api("extract", resp = resp))
#'
#' # Chat with failover:
#' resp2 <- gpt_api("chat_failover", messages = msgs)
#' cat(gpt_api("extract", resp = resp2))
#'
#' # Get token count:
#' cnt <- gpt_api("token_count", text = "This is a sample text.")
#' print(cnt)
#'
#' # Estimate cost:
#' cost_info <- gpt_api("estimate_cost", input_tokens = 1200, output_tokens = 800, model = "gpt-3.5-turbo")
#' print(cost_info)
#'
#' # Generate embeddings:
#' emb <- gpt_api("embedding", text = "OpenAI is great!")
#' print(emb)
#' }
NULL
