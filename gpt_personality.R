# Load required libraries
library(httr)
library(stringr)

###### NOTE: To use this code you also need to load the gpt_api() function: https://github.com/elkronos/openai_api/blob/main/gpt_api.R
###### This requires registering for your own API key which has an associated cost. See more information here: https://platform.openai.com/account/api-keys

#' Personality analysis using GPT
#'
#' This function receives a data frame and applies the OpenAI's GPT model to analyze personality traits. 
#' It takes as input a text (lyrics) and evaluates how the author of these lyrics would respond to certain statements 
#' regarding their personality. Each statement is evaluated in a scale from 1 (disagree strongly) to 5 (agree strongly). 
#' If the API doesn't return a valid value, it assigns NA to the respective column in the data frame.
#'
#' @param df A data frame containing a column named "Text" with the lyrics to be analyzed.
#'
#' @return The input data frame, with additional columns representing the personality analysis for each lyric.
#'
#' @examples
#' \dontrun{
#' library(httr)
#' library(stringr)
#'
# Create a data frame
#' df <- data.frame(ID = c(1, 2, 3, 4, 5, 6, 7), 
#'                  Text = c("(Turn around) Every now and then I get a little bit lonely And you’re never coming ’round (Turn around) Every now and then I get a little bit tired Of listening to the sound of my tears", 
#'                           "I love rock n’ roll So put another dime in the jukebox, baby I love rock n’ roll So come and take your time and dance with me Ow! ", 
#'                           "You think you’ve got it Oh, you think you’ve got it But got it just don’t get it when there’s nothin’ at all We get together Oh, we get together", 
#'                           "I don’t want no scrub, a scrub is a guy that can’t get no love from me",
#'                           "I see a little silhouetto of a man Scaramouch, scaramouch will you do the fandango Thunderbolt and lightning very very frightening me Gallileo, Gallileo, Gallileo, Gallileo, Gallileo Figaro",
#'                           "Hello darkness my old friend.",
#'                           "Look, if you had, one shot, or one opportunity to seize everything you ever wanted, in one moment, would you capture it, or just let it slip?"))
#' 
#' output <- gpt_personality(df)
#' print(output)
#' output[,c(2:12)] -> bfi_results
#' output_bfi <- score_BFI(bfi_results)
#' }
#'
#' @importFrom httr POST
#' @importFrom stringr str_extract
#' @export
gpt_personality <- function(df) {
  system_messages <- c("I see myself as someone who is reserved.", 
                       "I see myself as someone who is generally trusting.", 
                       "I see myself as someone who tends to be lazy.", 
                       "I see myself as someone who is relaxed, handles stress well.", 
                       "I see myself as someone who has few artistic interests.",
                       "I see myself as someone who is outgoing, sociable.",
                       "I see myself as someone who tends to find faults with others.",
                       "I see myself as someone who does a thorough job.",
                       "I see myself as someone who gets nervous easily.",
                       "I see myself as someone who has an active imagination.")
  
  for (i in 1:nrow(df)) {
    for (j in 1:length(system_messages)) {
      personality <- gpt_api(df$Text[i], 
                             system_message = paste("Given the following lyrics, how do you think the person who wrote this would respond to the following question:", 
                                                    "'", system_messages[j], 
                                                    "'. Please respond only with a [1] disagree strongly, [2] disagree a little, [3] Neither agree nor disagree, [4] Agree a little, [5] Agree strongly'."),
                             model = "gpt-3.5-turbo", 
                             temperature = 0.5, 
                             max_tokens = 1000, 
                             num_retries = 3, 
                             pause_base = 1, 
                             presence_penalty = 0.0, 
                             frequency_penalty = 0.0)
      
      # Filter the personality value
      personality_number <- as.integer(str_extract(personality, "1|2|3|4|5"))
      
      if (length(personality_number) == 0) {
        df[i, paste("bfi", j, sep = "_")] <- NA
      } else {
        df[i, paste("bfi", j, sep = "_")] <- personality_number
      }
    }
  }
  
  return(df)
}


# Load required libraries
library(httr)
library(stringr)
#' Score Big Five Inventory (BFI)
#'
#' This function calculates the Big Five Inventory (BFI) scores based on the personality analysis done by the function 
#' `gpt_personality_analysis`. It reverse scores relevant items and calculates the mean scores for each personality trait.
#' It also computes Cronbach's Alpha for each personality trait.
#'
#' @param df A data frame containing the personality analysis.
#'
#' @return A list containing: the data frame with the BFI scores, and Cronbach's Alpha for each personality trait.
#'
#' @examples
#' \dontrun{
#' library(httr)
#' library(stringr)
#'
#' df <- data.frame(ID = c(1, 2, 3, 4, 5, 6, 7), 
#'                  Text = c("Every now and then I get a little bit lonely...",
#'                           "I love rock n’ roll...",
#'                           "You think you’ve got it...",
#'                           "I don’t want no scrub...",
#'                           "I see a little silhouetto of a man...",
#'                           "Hello darkness my old friend.",
#'                           "Look, if you had, one shot..."))
#'
#' output <- gpt_personality_analysis(df)
#' print(output)
#' output[,c(2:12)] -> bfi_results
#' output_bfi <- score_BFI(bfi_results)
#'
#' print(output_bfi$df)
#' print(paste("Cronbach's Alpha for Extraversion: ", output_bfi$alpha_Extraversion))
#' print(paste("Cronbach's Alpha for Agreeableness: ", output_bfi$alpha_Agreeableness))
#' print(paste
score_BFI <- function(df) {
  # Reverse score relevant items
  df$bfi_1 <- 6 - df$bfi_1
  df$bfi_3 <- 6 - df$bfi_3
  df$bfi_4 <- 6 - df$bfi_4
  df$bfi_7 <- 6 - df$bfi_7
  df$bfi_5 <- 6 - df$bfi_5
  
  # Score BFI
  df$Extraversion <- rowMeans(df[, c("bfi_1", "bfi_6")])
  df$Agreeableness <- rowMeans(df[, c("bfi_2", "bfi_7")])
  df$Conscientiousness <- rowMeans(df[, c("bfi_3", "bfi_8")])
  df$Neuroticism <- rowMeans(df[, c("bfi_4", "bfi_9")])
  df$Openness_to_Experience <- rowMeans(df[, c("bfi_5", "bfi_10")])
  
  # Compute Cronbach's Alpha
  cronbachs_alpha_Extraversion <- psych::alpha(df[, c("bfi_1", "bfi_6")])$total$raw_alpha
  cronbachs_alpha_Agreeableness <- psych::alpha(df[, c("bfi_2", "bfi_7")])$total$raw_alpha
  cronbachs_alpha_Conscientiousness <- psych::alpha(df[, c("bfi_3", "bfi_8")])$total$raw_alpha
  cronbachs_alpha_Neuroticism <- psych::alpha(df[, c("bfi_4", "bfi_9")])$total$raw_alpha
  cronbachs_alpha_Openness_to_Experience <- psych::alpha(df[, c("bfi_5", "bfi_10")])$total$raw_alpha
  
  list(df = df, 
       alpha_Extraversion = cronbachs_alpha_Extraversion, 
       alpha_Agreeableness = cronbachs_alpha_Agreeableness, 
       alpha_Conscientiousness = cronbachs_alpha_Conscientiousness,
       alpha_Neuroticism = cronbachs_alpha_Neuroticism, 
       alpha_Openness_to_Experience = cronbachs_alpha_Openness_to_Experience)
}