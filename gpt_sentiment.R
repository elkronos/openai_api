# Load required libraries
library(httr)
library(stringr)

###### NOTE: To use this code you also need to load the gpt_api() function: https://github.com/elkronos/openai_api/blob/main/gpt_api.R
###### This requires registering for your own API key which has an associated cost. See more information here: https://platform.openai.com/account/api-keys

#' gpt_sentiment_analysis function
#'
#' @description This function conducts sentiment analysis on a given dataframe using the GPT-3.5-Turbo model from the OpenAI API. It operates by feeding each text in the dataframe to the GPT API, prompting it to assign a sentiment score of '-1' for negative, '0' for neutral, or '1' for positive. If the text is not related to the product, the API is instructed to say it's not related. 
#' If the sentiment extracted does not correspond to any of these, the function flags it as an "Invalid number". If there are multiple valid numbers, it's flagged as "Multiple valid numbers", otherwise, "Single valid number".
#' 
#' @param df The input dataframe, it must contain at least two columns: ID and Text.
#' 
#' @details 
#' \itemize{
#'  \item system_message: A specific instruction given to the API to ensure it understands the context and responds accordingly.
#'  \item model: Specifies the model to use, in this case 'gpt-3.5-turbo'.
#'  \item temperature: Controls randomness in the API's responses. Lower value makes the output more deterministic.
#'  \item max_tokens: Sets the maximum number of tokens in the output.
#'  \item num_retries: Specifies the number of retries to make in case the API call fails.
#'  \item pause_base: Specifies the base number of seconds to pause between retries.
#'  \item presence_penalty and frequency_penalty: These are scoring penalties to adjust the likelihood of tokens appearing in the output.
#' }
#' 
#' @return Returns a dataframe containing the original ID, the extracted sentiment and a flag indicating whether the sentiment extraction was successful or not.
#'
#' @examples
#' # Load required libraries
#' library(httr)
#' library(stringr)
#' 
#' # Create a data frame
#' df <- data.frame(ID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 
#'                  Text = c("I love this product!",
#'                           "The product is okay.",
#'                           "I hate this product!", 
#'                           "It's decent",
#'                           "I didn't care for it personally.",
#'                           "I love my cat!", 
#'                           "I hate that guys cat!",
#'                           "Rocks are a mineral.",
#'                           "It was similar to others I have used.",
#'                           "I don't know"))
#' 
#' # Use the function
#' output <- gpt_sentiment(df)
#' 
#' # View the output
#' print(output)
#' 
#' @importFrom httr RETRY
#' @importFrom stringr str_extract_all
#' @export
gpt_sentiment <- function(df) {
  sentiment_list <- list()
  flags <- vector()
  
  for (i in 1:nrow(df)) {
    sentiment <- gpt_api(df$Text[i], 
                         system_message = "You are an expert sentiment analyzer, analyzing emotion from text. Please carefully read the following text and respond with '-1' if the sentiment is negative, '0' if it's neutral, or '1' if it's positive.", 
                         model = "gpt-3.5-turbo", 
                         temperature = 0.2, 
                         max_tokens = 2000, 
                         num_retries = 3, 
                         pause_base = 1, 
                         presence_penalty = 0.0, 
                         frequency_penalty = 0.0)
    
    # Filter the sentiment value
    sentiment_numbers <- as.integer(str_extract_all(sentiment, "-1|0|1")[[1]])
    
    if (length(sentiment_numbers) == 0) {
      flags[i] <- "Invalid number"
      sentiment_list[[i]] <- NA
    } else {
      sentiment_list[[i]] <- sentiment_numbers
      flags[i] <- ifelse(length(sentiment_numbers) > 1, "Multiple valid numbers", "Single valid number")
    }
  }
  
  result <- data.frame(ID = df$ID, sentiment = unlist(sentiment_list), flag = flags)
  return(result)
}