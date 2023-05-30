#' This function sends a prompt to the OpenAI API and returns the generated text.
#' It handles API communication, error checking, and basic formatting of the response.
#'
#' Register for an API secert here: https://platform.openai.com/account/api-keys
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
#' @param api_key Character string for the API key assigned to the user by OpenAI
#'
#' @return A character string containing the generated text.
#'
#' @examples
#' \dontrun{
#' api_key <- Sys.getenv('OPENAI_API_KEY')
#' gpt_api("Write a poem about my cat Sam", api_key = api_key)
#' gpt_api("Tell me a joke.", model = "gpt-3.5-turbo", api_key = api_key)
#' gpt_api("Write a poem about my cat Sam", model = "gpt-4", api_key = api_key)
#' }
#' @export
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50,
                system_message = NULL, num_retries = 3, pause_base = 1,
                presence_penalty = 0.0, frequency_penalty = 0.0, api_key) {
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
    # prepend system message to messages list
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }

  response <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", api_key)),
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

#' Estimate token count of a text string
#'
#' This function provides a rough estimate of the number of tokens in a text string,
#' based on splitting the text into words, punctuation and whitespaces.
#' Note that this is an approximation and may not match the token count used
#' by specific language models.
#'
#' @param text A character string for which to estimate the token count.
#'
#' @return An integer giving the estimated number of tokens in the input text.
#'
#' @examples
#' count_tokens("Hello, world!")
#' count_tokens("This is a longer sentence with more tokens.")
#'
#' @export
count_tokens <- function(text) {
  # Split text into words, punctuation and whitespaces
  tokens <- strsplit(text, "(?<=\\W)(?=\\w)|(?<=\\w)(?=\\W)", perl = TRUE)[[1]]

  # Return number of tokens
  return(length(tokens))
}

#' Estimate the cost based on input and output tokens
#'
#' This function calculates the estimated cost based on the number of input and output tokens and the specified model.
#'
#' @param input_tokens The number of input tokens.
#' @param output_tokens The number of output tokens.
#' @param model The model to use for estimation (default is "gpt-3.5-turbo").
#'
#' @return A list with the following elements:
#'   \describe{
#'     \item{input_tokens}{The number of input tokens.}
#'     \item{output_tokens}{The number of output tokens.}
#'     \item{input_cost}{The estimated cost for the input tokens.}
#'     \item{output_cost}{The estimated cost for the output tokens.}
#'     \item{total_cost}{The total estimated cost (input_cost + output_cost).}
#'   }
#'
#' @export
#'
#' @examples
#' input_tokens <- 1200
#' output_tokens <- 800
#' model <- "gpt-4"
#'
#' cost_estimate <- estimate_cost(input_tokens, output_tokens, model)
#' print(paste("Input tokens:", cost_estimate$input_tokens))
#' print(paste("Output tokens:", cost_estimate$output_tokens))
#' print(paste("Input cost:", cost_estimate$input_cost))
#' print(paste("Output cost:", cost_estimate$output_cost))
#' print(paste("Total cost:", cost_estimate$total_cost))
#'
estimate_cost <- function(input_tokens, output_tokens, model = "gpt-3.5-turbo") {

  if (model == "gpt-4") {
    price_1k_in <- ifelse(input_tokens <= 8000, 0.03, 0.06)
    price_1k_out <- ifelse(output_tokens <= 8000, 0.06, 0.12)
  } else {
    price_1k_in <- 0.002
    price_1k_out <- 0.002
  }

  input_cost <- input_tokens / 1000 * price_1k_in
  output_cost <- output_tokens / 1000 * price_1k_out
  total_cost <- input_cost + output_cost

  result <- list(
    input_tokens = input_tokens,
    output_tokens = output_tokens,
    input_cost = input_cost,
    output_cost = output_cost,
    total_cost = total_cost
  )

  return(result)
}
