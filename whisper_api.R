library(httr)
library(jsonlite)
#' Transcribe Audio Using OpenAI's API
#'
#' This function uses the OpenAI API to transcribe an audio file.
#'
#' @param file_path Character. The file path to the audio file to be transcribed.
#' @param api_key Character. The OpenAI API key.
#' @param model Character. The name of the OpenAI model to use for transcription. Default is 'whisper-1'.
#' @param response_format Character. The format of the response from the API. Default is 'json'.
#' @param temperature Numeric. The temperature value to use in the transcription model. Default is 0.
#' @param language Character. The language of the audio file. Default is NULL.
#' @param prompt Character. The prompt to provide to the transcription model. Default is NULL.
#'
#' @return A list containing the parsed response from the API in the specified response format.
#'
#' @importFrom httr POST add_headers content
#' @importFrom jsonlite fromJSON
#'
#' @examples
#' \dontrun{
#' api_key <- "your-api-key"
#' file_path <- "C:/path_to_your_file/audiofile.wav"
#' transcription <- whisper_transcribe(file_path, api_key)
#' print(transcription)
#' 
#' # You may download an example audio file to transcribe here
#' # https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0061_8k.wav
#' }
#'
#' @note The `api_key` is sensitive information, do not share it or publish it. Make sure to store it securely.
#' The `file_path` should be a valid path to a .wav file.
#'
#' @seealso \url{https://openai.com/research/whisper/} for more information on the Whisper ASR API.
#'
#' @export
whisper_transcribe <- function(file_path, api_key, model = "whisper-1", response_format = "json", temperature = 0, language = NULL, prompt = NULL) {
  url <- "https://api.openai.com/v1/audio/transcriptions"
  body <- list(
    file = upload_file(file_path),
    model = model,
    response_format = response_format,
    temperature = temperature
  )
  
  if (!is.null(language)) {
    body$language <- language
  }
  
  if (!is.null(prompt)) {
    body$prompt <- prompt
  }
  
  res <- POST(url,
              add_headers(c(Authorization = paste("Bearer", api_key),
                            "Content-Type" = "multipart/form-data")),
              body = body,
              encode = "multipart")
  
  content(res, "parsed", "application/json")
}


#' Translate Audio Using OpenAI's API
#'
#' This function uses the OpenAI API to translate an audio file.
#'
#' @param file_path Character. The file path to the audio file to be translated.
#' @param api_key Character. The OpenAI API key.
#' @param model Character. The name of the OpenAI model to use for translation. Default is 'whisper-1'.
#' @param response_format Character. The format of the response from the API. Default is 'json'.
#' @param temperature Numeric. The temperature value to use in the translation model. Default is 0.
#' @param prompt Character. The prompt to provide to the translation model. Default is NULL.
#'
#' @return A list containing the parsed response from the API in the specified response format.
#'
#' @importFrom httr POST add_headers content
#' @importFrom jsonlite fromJSON
#'
#' @examples
#' \dontrun{
#' api_key <- "your-api-key"
#' file_path <- "C:/path_to_your_file/audiofile.wav"
#' translation <- whisper_translate(file_path, api_key)
#' print(translation)
#' }
#'
#' @note The `api_key` is sensitive information, do not share it or publish it. Make sure to store it securely.
#' The `file_path` should be a valid path to a .wav file.
#'
#' @seealso \url{https://openai.com/research/whisper/} for more information on the Whisper ASR API.
#'
#' @export
whisper_translate <- function(file_path, api_key, model = "whisper-1", response_format = "json", temperature = 0, prompt = NULL) {
  url <- "https://api.openai.com/v1/audio/translations"
  body <- list(
    file = upload_file(file_path),
    model = model,
    response_format = response_format,
    temperature = temperature
  )
  
  if (!is.null(prompt)) {
    body$prompt <- prompt
  }
  
  res <- POST(url,
              add_headers(c(Authorization = paste("Bearer", api_key),
                            "Content-Type" = "multipart/form-data")),
              body = body,
              encode = "multipart")
  
  content(res, "parsed", "application/json")
}