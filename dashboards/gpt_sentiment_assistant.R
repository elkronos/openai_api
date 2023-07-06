# Load packages
library(shiny)
library(DT)
library(plotly)
library(httr)
library(dplyr)
library(readr)
library(stringr)
library(future)
library(shinybusy)

## Make example data
## Variables must be named the same in your data as in the example output  - working on making that flexible. 
# df <- data.frame(
#   ID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#   Text = c("I love this product!",
#            "The product is okay.",
#            "I hate this product!",
#            "It's decent",
#            "I didn't care for it personally.",
#            "I love my cat!",
#            "I hate that guy's cat!",
#            "Rocks are a mineral.",
#            "It was similar to others I have used.",
#            "I don't know"),
#   group = c(rep("Group A", 5), rep("Group B", 5))
# )
# # Write the dataframe to a CSV file
# write.csv(df, "text_data.csv", row.names = FALSE)

# API Function
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50, 
                    system_message = NULL, num_retries = 3, pause_base = 1, 
                    presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }
  
  response <- RETRY(
    "POST",
    url = "https://api.openai.com/v1/chat/completions", 
    add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY"))),
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
  
  stop_for_status(response)
  
  if (length(content(response)$choices) > 0) {
    message <- content(response)$choices[[1]]$message$content
  } else {
    message <- "The model did not return a message. You may need to increase max_tokens."
  }
  
  clean_message <- gsub("\n", " ", message)
  clean_message <- str_trim(clean_message)
  return(clean_message)
}

# Sentiment Function
gpt_sentiment <- function(df, system_message = "You are an expert sentiment analyzer, analyzing emotion from text. Please carefully read the following text and respond with '-1' if the sentiment is negative, '0' if it's neutral, or '1' if it's positive.", temperature = 0.2) {
  if (nrow(df) < 1 || ncol(df) < 2) {
    stop("The data frame must have at least one row and two columns")
  }
  
  sentiment_list <- list()
  flags <- vector()
  
  for (i in 1:nrow(df)) {
    sentiment <- gpt_api(as.character(df[,2][i]), 
                         system_message = system_message, 
                         model = "gpt-3.5-turbo", 
                         temperature = temperature, 
                         max_tokens = 2000, 
                         num_retries = 3, 
                         pause_base = 1, 
                         presence_penalty = 0.0, 
                         frequency_penalty = 0.0)
    
    sentiment_numbers <- as.integer(str_extract_all(sentiment, "-1|0|1")[[1]])
    
    if (length(sentiment_numbers) == 0) {
      flags[i] <- "Invalid number"
      sentiment_list[[i]] <- NA
    } else {
      sentiment_list[[i]] <- sentiment_numbers
      flags[i] <- ifelse(length(sentiment_numbers) > 1, "Multiple valid numbers", "Single valid number")
    }
  }
  
  result <- data.frame(ID = df$ID, Text = df[,2], Sentiment = unlist(sentiment_list), Flag = flags)
  return(result)
}

# User Interface
ui <- fluidPage(
  shinybusy::add_busy_spinner(spin = "cube-grid"),  # Add the loading spinner
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Choose CSV File", accept = ".csv"),
      textAreaInput("system_message", "System Message", 
                    value = "You are an expert sentiment analyzer, analyzing emotion from text. Please carefully read the following text and respond with '-1' if the sentiment is negative, '0' if it's neutral, or '1' if it's positive.",
                    width = "100%"),
      numericInput("temperature", "Temperature", value = 0.5, min = 0, max = 1, step = 0.1),
      actionButton("run", "Run Analysis and Save Data")
    ),
    mainPanel(
      plotlyOutput("barplot"),
      DT::dataTableOutput("table")
    )
  )
)

# Server function
server <- function(input, output, session) {
  # Read the data from the file
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath, stringsAsFactors = FALSE)
  })
  
  # Perform sentiment analysis on each row of data
  sentiment_result <- reactive({
    req(data())
    gpt_sentiment(data(), system_message = input$system_message, temperature = input$temperature)
  })
  
  # Update outputs based on sentiment analysis result
  observeEvent(input$run, {
    req(sentiment_result())
    
    # Save the output data
    file_name <- paste0("output_", Sys.Date(), "_", format(Sys.time(), "%H%M%S"), ".csv")
    write.csv(sentiment_result(), file = file_name, row.names = FALSE)
    
    # Display data in DT::datatable
    output$table <- DT::renderDataTable(sentiment_result())
    
    # Display sentiment counts in bar plot
    sentiment_counts <- table(sentiment_result()$Sentiment, useNA = "no")
    plot_data <- data.frame(sentiment = names(sentiment_counts), count = as.integer(sentiment_counts))
    total <- sum(plot_data$count)
    plot_data$percentage <- round((plot_data$count / total) * 100, 2)
    output$barplot <- renderPlotly({
      plot_ly(plot_data, x = ~sentiment, y = ~count, type = 'bar', text = ~paste0(percentage, "%"), textposition = 'auto') %>%
        layout(title = "Sentiment Distribution", 
               xaxis = list(title = "Sentiment"), 
               yaxis = list(title = "Count"),
               margin = list(t = 75))  # Increase top margin to avoid crowding
    })
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
