# Load libraries
library(httr)
library(jsonlite)

# Define OpenAI API Key and Base URL
initialize_openai_api <- function(api_key) {
  options(openai_api_key = api_key)
  options(openai_base_url = "https://api.openai.com/v1")
}

# Set up headers
get_headers <- function() {
  add_headers(
    `Content-Type` = "application/json",
    Authorization = paste("Bearer", getOption("openai_api_key"))
  )
}

# Upload File for Fine-Tuning
upload_file <- function(file_path) {
  upload_url <- paste0(getOption("openai_base_url"), "/files")
  
  res <- POST(
    url = upload_url,
    get_headers(),
    body = list(file = upload_file(file_path)),
    encode = "multipart"
  )
  
  content(res)
}

# Create Fine-Tuning Job
create_fine_tuning_job <- function(training_file_id, model = "gpt-3.5-turbo") {
  fine_tuning_url <- paste0(getOption("openai_base_url"), "/fine-tuning/jobs")
  
  body <- toJSON(list(
    training_file = training_file_id,
    model = model
  ))
  
  res <- POST(
    url = fine_tuning_url,
    get_headers(),
    body = body,
    encode = "json"
  )
  
  content(res)
}

# List Fine-Tuning Jobs
list_fine_tuning_jobs <- function() {
  list_url <- paste0(getOption("openai_base_url"), "/fine-tuning/jobs")
  
  res <- GET(
    url = list_url,
    get_headers()
  )
  
  content(res)
}

# Cancel Fine-Tuning Job
cancel_fine_tuning_job <- function(job_id) {
  cancel_url <- paste0(getOption("openai_base_url"), "/fine-tuning/jobs/", job_id)
  
  res <- DELETE(
    url = cancel_url,
    get_headers()
  )
  
  content(res)
}

# Check Status of a Fine-Tuning Job
check_finetuning_status <- function(job_id) {
  status_url <- paste0(getOption("openai_base_url"), "/fine-tuning/jobs/", job_id)
  
  res <- GET(
    url = status_url,
    get_headers()
  )
  
  content(res)
}

#' # Individual example use
#' 
#' # Initialize OpenAI API
#' initialize_openai_api("your_api_key")
#' 
#' # Upload a file
#' file_info <- upload_file("path_to_your_file.jsonl")
#' print(file_info)
#' 
#' # Create a fine-tuning job
#' job_info <- create_fine_tuning_job(file_info$id)
#' print(job_info)
#' 
#' # List fine-tuning jobs
#' jobs_info <- list_fine_tuning_jobs()
#' print(jobs_info)
#' 
#' # Check status of a fine-tuning job
#' status_info <- check_finetuning_status(job_info$id)
#' print(status_info)
#' 
#' # Cancel a fine-tuning job (if needed)
#' # cancel_info <- cancel_fine_tuning_job(job_info$id)
#' # print(cancel_info)

# Wrapper function
gpt_finetune <- function(operation, ...) {
  # Initialize OpenAI API (if not already done)
  if (is.null(getOption("openai_api_key"))) {
    stop("Please initialize OpenAI API by calling initialize_openai_api(api_key)")
  }
  
  switch(operation,
         upload_file = {
           args <- list(...)
           upload_file(args$file_path)
         },
         create_job = {
           args <- list(...)
           create_fine_tuning_job(args$training_file_id, args$model)
         },
         list_jobs = {
           list_fine_tuning_jobs()
         },
         check_status = {
           args <- list(...)
           check_finetuning_status(args$job_id)
         },
         cancel_job = {
           args <- list(...)
           cancel_fine_tuning_job(args$job_id)
         },
         stop("Invalid operation specified")
  )
}

#' # Wrapper example usage
#' 
#' # Initialize OpenAI API
#' initialize_openai_api("your_api_key")
#' 
#' # Upload a file
#' file_info <- gpt_finetune("upload_file", file_path = "path_to_your_file.jsonl")
#' print(file_info)
#' 
#' # Create a fine-tuning job
#' job_info <- gpt_finetune("create_job", training_file_id = file_info$id, model = "gpt-3.5-turbo")
#' print(job_info)
#' 
#' # List fine-tuning jobs
#' jobs_info <- gpt_finetune("list_jobs")
#' print(jobs_info)
#' 
#' # Check status of a fine-tuning job
#' status_info <- gpt_finetune("check_status", job_id = job_info$id)
#' print(status_info)
#' 
#' # Cancel a fine-tuning job (if needed)
#' # cancel_info <- gpt_finetune("cancel_job", job_id = job_info$id)
#' # print(cancel_info)