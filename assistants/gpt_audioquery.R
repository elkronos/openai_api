#' Transcribe and process recorded audio using OpenAI's Whisper and GPT APIs
#'
#' This function records audio for a given duration, transcribes it using the
#' Whisper API, and sends the transcription to the GPT API for further processing.
#' 
#' You should load the GPT and Whisper functions in this repo in order to use this function.
#' 
#' GPT: https://github.com/elkronos/openai_api/blob/main/gpt_api.R
#' Whisper: https://github.com/elkronos/openai_api/blob/main/whisper_api.R
#'
#' @param record_time Numeric, the duration (in seconds) for which to record audio.
#'   Must be a positive number.
#' @param api_key Character, the API key for OpenAI. This key is used for both the
#'   Whisper and GPT API calls.
#' @param language Character, optional. The language of the audio to be transcribed.
#'   If NULL (default), the Whisper API will attempt to automatically determine the language.
#' @param prompt Character, optional. An additional prompt that can be provided to the Whisper API.
#'
#' @return Character, the response from the GPT API.
#'
#' @examples
#' \dontrun{
#' response <- whisper_to_gpt(10, "your_api_key")
#' print(response)
#' }
#'
#' @seealso
#' \url{https://openai.com/research/whisper/} for more information on the Whisper API.
#' \url{https://openai.com/research/gpt-3.5-turbo/} for more information on the GPT-3.5-turbo API.
#'
#' @export
whisper_to_gpt <- function(record_time, api_key, language = NULL, prompt = NULL) {
  
  # Ensure record_time is numeric and greater than 0
  if(!is.numeric(record_time) || record_time <= 0) {
    stop("Error: record_time must be a positive number.")
  }
  
  # Ensure api_key is not empty
  if(!nzchar(api_key)) {
    stop("Error: API key must not be empty.")
  }
  
  # Record audio for a specified duration
  rec <- audio::record()
  Sys.sleep(record_time)
  audio_file <- "audio.wav"
  audio::stop(rec, audio_file)
  
  # Transcribe the audio using Whisper API
  transcript <- tryCatch(
    {
      whisper_transcribe(audio_file, api_key, language = language, prompt = prompt)
    },
    error = function(e) {
      stop("Error in whisper_transcribe: ", e$message)
    }
  )
  
  # Check the status of transcription and retrieve the result
  if (transcript$status == "completed") {
    message <- transcript$alternatives[[1]]$transcript
  } else {
    stop("Transcription failed. Status: ", transcript$status)
  }
  
  # Send the transcription to GPT API and get the response
  gpt_response <- tryCatch(
    {
      gpt_api(message)
    },
    error = function(e) {
      stop("Error in gpt_api: ", e$message)
    }
  )
  
  # Return GPT's response
  return(gpt_response)
}