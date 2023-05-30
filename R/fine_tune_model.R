#' Fine-tunes a GPT model using the OpenAI API
#'
#' This function allows you to fine-tune a GPT model using the OpenAI API. The model is trained on a provided dataset and can be configured with several parameters.
#' This function makes use of the `httr` and `jsonlite` libraries to send and handle HTTP requests.
#'
#' @param training_file (character) The ID of the file to use for training. This should be a string that represents the unique identifier for the file in the OpenAI system.
#' @param validation_file (character) (Optional) The ID of the file to use for validation. This should also be a string that represents the unique identifier for the file in the OpenAI system. If not provided, the training file will be used for validation as well.
#' @param model (character) (Optional) The name of the model to use for fine-tuning. This should be a string that represents one of the model names supported by OpenAI (e.g., "curie"). Default is 'curie'.
#' @param n_epochs (numeric) (Optional) The number of epochs for training. This should be an integer value. Default is 4.
#' @param batch_size (numeric) (Optional) The size of the batches used in training. This should be an integer value. If not provided, the API will choose a size automatically.
#' @param learning_rate_multiplier (numeric) (Optional) A multiplier for the learning rate. This should be a numeric value. If not provided, the API will choose a value automatically.
#' @param prompt_loss_weight (numeric) (Optional) The weight of the prompt loss in the loss function. This should be a numeric value. Default is 0.01.
#' @param compute_classification_metrics (logical) (Optional) Whether to compute classification metrics during training. This should be a boolean value (TRUE/FALSE). Default is FALSE.
#' @param classification_n_classes (numeric) (Optional) The number of classes for classification. This should be an integer value. Required if compute_classification_metrics is TRUE.
#' @param classification_positive_class (character) (Optional) The positive class for classification. This should be a string that represents the class label. Required if compute_classification_metrics is TRUE and classification_n_classes is 2.
#' @param classification_betas (numeric) (Optional) The betas for the Adam optimizer. This should be a numeric value.
#' @param suffix (character) (Optional) A suffix for the name of the fine-tuned model. This should be a string that will be appended to the name of the fine-tuned model.
#'
#' @return A list representing the response from the OpenAI API. If the request is successful, this list will contain information about the fine-tuned model, such as its name and status. If the request is unsuccessful, an error will be thrown with the status code from the API.
#' @importFrom httr POST add_headers content
#' @importFrom jsonlite toJSON
#' @export
#'
#' @examples
#' \dontrun{
#' training_file_id <- "your_training_file_id"
#' validation_file_id <- "your_validation_file_id"
#' your_api_key <- "your_api_key"
#'
#' result <- fine_tune_model(
#'   training_file = training_file_id,
#'   validation_file = validation_file_id,
#'   model = "curie",
#'   n_epochs = 4,
#'   batch_size = NULL,
#'   learning_rate_multiplier = NULL,
#'   prompt_loss_weight = 0.01,
#'   compute_classification_metrics = FALSE,
#'   classification_n_classes = NULL,
#'   classification_positive_class = NULL,
#'   classification_betas = NULL,
#'   suffix = NULL
#' )
#' print(result)
#' }
# Define function
fine_tune_model <- function(training_file, validation_file=NULL, model="curie",
                            n_epochs=4, batch_size=NULL, learning_rate_multiplier=NULL,
                            prompt_loss_weight=0.01, compute_classification_metrics=FALSE,
                            classification_n_classes=NULL, classification_positive_class=NULL,
                            classification_betas=NULL, suffix=NULL) {

  # Define the API URL
  url <- "https://api.openai.com/v1/fine-tunes"

  # Create the request body
  body <- list(
    training_file = training_file,
    validation_file = validation_file,
    model = model,
    n_epochs = n_epochs,
    batch_size = batch_size,
    learning_rate_multiplier = learning_rate_multiplier,
    prompt_loss_weight = prompt_loss_weight,
    compute_classification_metrics = compute_classification_metrics,
    classification_n_classes = classification_n_classes,
    classification_positive_class = classification_positive_class,
    classification_betas = classification_betas,
    suffix = suffix
  )

  # Remove NULL entries
  body <- body[!unlist(lapply(body, is.null))]

  # Define the headers
  headers <- add_headers(
    "Authorization" = paste("Bearer", "your_api_key"),
    "Content-Type" = "application/json"
  )

  # Make the POST request
  response <- POST(url, headers, body = jsonlite::toJSON(body, auto_unbox = TRUE))

  # Check if the request was successful
  if (response$status_code >= 200 && response$status_code < 300) {
    # If the request was successful, parse and return the response
    return(content(response, "parsed"))
  } else {
    # If the request was unsuccessful, throw an error
    stop(paste("Request failed with status code", response$status_code))
  }
}
