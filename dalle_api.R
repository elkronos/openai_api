# Load packages
library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)
#' API Call to OpenAI
#'
#' This function is a common handler for making API calls to OpenAI. It was primarily made to be used with the other functions in this script.
#'
#' @param endpoint A string specifying the API endpoint. This should not include the base URL, as it is automatically prepended.
#' @param method A string specifying the HTTP method for the API call, either "GET" or "POST".
#' @param headers A named list of HTTP headers to include in the request.
#' @param body An optional argument. If the HTTP method is "POST", this should be a list that will be included in the request body.
#' @param query An optional argument. If provided, it should be a list to be included as query parameters in the URL.
#'
#' @return The function returns the content of the API response as a list. If the server response is not a JSON, it throws an error.
#'
#' @importFrom httr GET POST add_headers content http_type
#' @importFrom jsonlite fromJSON
#' @importFrom dplyr %>% 
#'
#' @examples
#' # Example of a GET request:
#' headers <- c("Authorization" = paste0("Bearer ", api_key))
#' response <- dalle_call("engines", "GET", headers)
#'
#' # Example of a POST request:
#' headers <- c("Authorization" = paste0("Bearer ", api_key),
#'              "Content-Type" = "application/json")
#' body <- list(prompt = "Translate the following English text to French: '{}'",
#'              max_tokens = 60)
#' response <- dalle_call("engines/davinci-codex/completions", "POST", headers, body)
#' 
#' @export
dalle_call <- function(endpoint, method, headers, body = NULL, query = NULL) {
  base_url <- "https://api.openai.com/v1/"
  
  if (method == "GET") {
    response <- GET(paste0(base_url, endpoint), add_headers(headers), query = query)
  } else if (method == "POST") {
    response <- POST(paste0(base_url, endpoint), add_headers(headers), body = body, encode = "json")
  }
  
  if (http_type(response) == "application/json") {
    content(response, as = "text", encoding = "UTF-8") %>% fromJSON()
  } else {
    stop("Error: Unexpected response from server")
  }
}

#' dalle_generate
#'
#' This function handles API calls to generate and download images based on given prompts. 
#' It uses a provided API key to authenticate and then makes a POST request to the "images/generations" endpoint.
#' If the API call is successful and the returned data contains image URLs, it downloads the images and saves them to the provided location. 
#' It also updates a global dataframe, `prompts_df`, with the prompt and the timestamp.
#'
#' @param api_key A string containing the API key for authentication.
#' @param prompt A string that serves as the prompt based on which the image will be generated.
#' @param n An integer indicating the number of images to generate. Defaults to 1.
#' @param size A string defining the size of the generated image in the format "widthxheight". Defaults to "1024x1024".
#' @param path_to_save A string specifying the path where the images should be saved. Defaults to "C:/Users/JChas/OneDrive/Desktop/repo_functions".
#'
#' @return A list containing the result of the API call. If the API call is successful and contains URLs of generated images, those images are also downloaded and saved to the specified path.
#'
#' @importFrom utils download.file
#' @importFrom base paste Sys.time format file.path rbind data.frame
#' @importFrom grDevices png
#' @importFrom stats setNames
#'
#' @examples
#' \dontrun{
#' api_key <- "your_api_key_here"
#' prompt <- "A sunset over the mountains"
#' n <- 2
#' size <- "1024x1024"
#' path_to_save <- "C:/your/path"
#' result <- dalle_generate(api_key, prompt, n, size, path_to_save)
#' }
#' @export
dalle_generate <- function(api_key, prompt, n = 1, size = "1024x1024", path_to_save = "C:/your/path") {
  headers <- c("Content-Type" = "application/json",
               "Authorization" = paste("Bearer", api_key))
  body <- list("prompt" = prompt, "n" = n, "size" = size)
  
  result <- api_call("images/generations", "POST", headers, body)
  
  if ("data" %in% names(result)) {
    if ("url" %in% colnames(result$data)) {
      for (i in seq_along(result$data$url)) {
        timestamp <- format(Sys.time(), "%Y%m%d%H%M%S") # get timestamp
        url <- result$data$url[i]
        download.file(url, destfile = file.path(path_to_save, paste0("image_", timestamp, "_", i, ".png")), mode = "wb") # add timestamp to image name
      }
      # append the prompt to the dataframe
      prompts_df <<- rbind(prompts_df, data.frame(Prompt = prompt, Timestamp = timestamp))
    } else {
      stop("Error: Unexpected data structure received from server")
    }
  }
  
  return(result)
}

#' Save Prompts DataFrame to a CSV File
#'
#' This function saves a dataframe `prompts_df` to a specified CSV file.
#'
#' @param path_to_save A character string representing the path where the CSV file will be saved. 
#' Default is "C:/Users/JChas/OneDrive/Desktop/repo_functions/prompts.csv".
#'
#' @return Invisible NULL. The function is called for its side effect: it writes to a file on disk.
#' 
#' @importFrom utils write.csv
#' 
#' @examples
#' # Assume prompts_df is available
#' # save_prompts()
#' # save_prompts("path/to/save.csv")
#'
#' @seealso \code{\link[utils]{write.csv}} for the underlying writing function.
#' 
#' @export
save_prompts <- function(path_to_save = "C:/Users/JChas/OneDrive/Desktop/repo_functions/prompts.csv") {
  write.csv(prompts_df, file = path_to_save, row.names = FALSE)
}

#' Generate Variations
#'
#' This function generates variations of an image by applying transformations such as scaling, modulating brightness, and applying blur. The resulting image is saved to disk.
#'
#' @param image_url The URL of the image to be processed.
#' @param save_path The path where the resulting image will be saved. Default is "C:/Users/JChas/OneDrive/Desktop/repo_functions/".
#' @param scale_parameter The scale parameter for image scaling. Default is "50%".
#' @param brightness_parameter The brightness parameter for modulating brightness. Default is 80.
#' @param radius_parameter The radius parameter for applying blur. Default is 0.
#' @param sigma_parameter The sigma parameter for applying blur. Default is 1.
#'
#' @importFrom magick image_read image_scale image_modulate image_blur image_write
#'
#' @examples
#' # Generate variations of an image from a URL
#' dalle_variations("https://example.com/image.jpg")
#' 
#' # Generate variation from what you queried using the dalle_api() function
#' dalle_variations(result$data$url[1])
#'
#' @export
dalle_variation <- function(image_url,
                            save_path = "C:/your/path",
                            scale_parameter = "50%",
                            brightness_parameter = 80, 
                            radius_parameter = 0,
                            sigma_parameter = 1) {
  # Read the image from the URL
  image <- image_read(image_url)
  
  # Apply transformations
  image <- image %>% 
    image_scale(scale_parameter) %>%  # Reduce size
    image_modulate(brightness_parameter) %>%  # Reduce brightness
    image_blur(radius_parameter, sigma_parameter)  # Apply blur
  
  # Get the current date and time
  timestamp <- Sys.time()
  
  # Format the timestamp to use in a filename
  timestamp_str <- format(timestamp, "%Y%m%d_%H%M%S")
  
  # Construct the filename with the timestamp
  filename <- paste0(save_path, "image_", timestamp_str, ".png")
  
  # Save the image to disk
  image_write(image, path = filename)
}