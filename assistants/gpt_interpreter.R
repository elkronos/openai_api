# Load packages
library(httr)
library(stringr)
library(dplyr)
library(stringr)

# Chunk Text into Smaller Fragments
chunk_text <- function(text) {
  # Split text into words
  tokens <- unlist(strsplit(text, "\\s+"))
  # Chunk tokens to have approximately 3000 tokens per chunk
  token_chunks <- split(tokens, ceiling(seq_along(tokens)/3000))
  # Combine tokens back into text chunks
  text_chunks <- lapply(token_chunks, paste, collapse = " ")
  
  return(text_chunks)
}

# Function to interact with OpenAI API
gpt_api <- function(prompt, model = "gpt-3.5-turbo", temperature = 0.1, max_tokens = 50, 
                    system_message = NULL, num_retries = 3, pause_base = 5, # Increased pause_base
                    presence_penalty = 0.0, frequency_penalty = 0.0) {
  
  messages <- list(list(role = "user", content = prompt))
  if (!is.null(system_message)) {
    messages <- append(list(list(role = "system", content = system_message)), messages)
  }
  
  response <- NULL
  tryCatch({
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
  }, error = function(e) {
    cat("Error during API call: ", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(response) && length(content(response)$choices) > 0) {
    message <- content(response)$choices[[1]]$message$content
  } else {
    message <- "The model did not return a message or an error occurred."
  }
  
  clean_message <- gsub("\n", " ", message)
  clean_message <- str_trim(clean_message)
  
  return(clean_message)
}

# Send Text Chunks to GPT for Stance Determination
send_to_gpt_for_stance <- function(text_chunk) {
  prompt <- paste("Read the following comment about the FTC's proposed non-compete ban and determine whether or not the comment is for, against the ban, neutral, or whether the text makes it unknown. Only respond with a [1] if in favor of the rule or against noncompetes, [2] if against the rule or is favorable toward noncompetes, [3] if equally neutral about noncompetes, or [4] if it is unknown how they feel about noncompetes or the proposed rule.", text_chunk)
  system_message <- "You have 20 years of experience reviewing FTC comments and are tasked with determining the stance on a comment. You will only respond with a [1] if in favor of the rule or against noncompetes, [2] if against the rule or is favorable toward noncompetes, [3] if equally neutral about noncompetes, or [4] if it is unknown how they feel about noncompetes or the proposed rule."
  
  response <- gpt_api(prompt, system_message = system_message, max_tokens = 50)
  return(response)
}

# Function for Sampling Data
sample_data <- function(df, sample_size, strata_column = NULL, text_column = "Comment") {
  set.seed(123)
  if (!is.null(strata_column)) {
    if (!strata_column %in% names(df)) {
      stop("Strata column not found in the dataframe")
    }
    
    strata_levels <- unique(df[[strata_column]])
    num_strata <- length(strata_levels)
    sample_per_strata <- sample_size %/% num_strata
    extra_samples <- sample_size %% num_strata
    
    sampled_dfs <- lapply(strata_levels, function(level) {
      filtered_data <- df %>% filter(.[[strata_column]] == level, !is.na(.[[text_column]]))
      if (nrow(filtered_data) < sample_per_strata) {
        return(filtered_data)
      } else {
        return(sample_n(filtered_data, sample_per_strata))
      }
    })
    
    sampled_df <- do.call(rbind, sampled_dfs)
    remaining_samples <- sample_size - nrow(sampled_df)
    if (remaining_samples > 0) {
      additional_samples <- df %>% filter(!is.na(.[[text_column]])) %>% sample_n(remaining_samples)
      sampled_df <- rbind(sampled_df, additional_samples)
    }
    
    return(sampled_df)
  } else {
    df %>% filter(!is.na(.[[text_column]])) %>% sample_n(min(nrow(.), sample_size))
  }
}

# Function to Process Comments and Update Data Frame
gpt_interpret <- function(df, text_column = "Comment", max_tokens_per_chunk = 3000, delay_seconds = 1.5) {
  # Ensure the text column exists
  if (!text_column %in% names(df)) {
    stop("Text column not found in the dataframe")
  }
  
  # Initialize columns for stance and flag
  df$stance <- NA
  df$flag <- NA
  
  # Apply chunking and send to GPT
  for (i in seq_len(nrow(df))) {
    text_chunks <- chunk_text(df[[text_column]][i])
    
    if (length(text_chunks) > 1 || nchar(text_chunks[[1]]) > max_tokens_per_chunk) {
      df$flag[i] <- "Text too long or multiple chunks"
      next
    }
    
    stance_list <- vector("list", length(text_chunks))
    
    for (j in seq_along(text_chunks)) {
      response <- send_to_gpt_for_stance(text_chunks[[j]])
      if (is.null(response) || response == "The model did not return a message or an error occurred.") {
        df$flag[i] <- "API call failed"
        next
      }
      stance_numbers <- as.integer(str_extract_all(response, "1|2|3|4")[[1]])
      
      if (length(stance_numbers) == 0) {
        df$flag[i] <- "Invalid number"
        stance_list[[j]] <- NA
      } else {
        stance_list[[j]] <- stance_numbers
        df$flag[i] <- ifelse(length(stance_numbers) > 1, "Multiple valid numbers", "Single valid number")
      }
      
      # Pause after each successful API call
      Sys.sleep(delay_seconds)
    }
    
    df$stance[i] <- paste(unlist(stance_list), collapse = "; ")
    
    # Progress message for every processed text
    cat("Processed", i, "out of", nrow(df), "texts.\n")
  }
  
  return(df)
}

# Wrapper Function
process_and_sample <- function(df, sample_size, strata_column = NULL, text_column = "Comment") {
  sampled_df <- sample_data(df, sample_size, strata_column, text_column)
  processed_df <- gpt_interpret(sampled_df, text_column)
  return(processed_df)
}

# Clean state column for data cleaning in this example
identify_states <- function(df, column_name) {
  # Ensure the column exists in the data frame
  if (!column_name %in% names(df)) {
    stop("Column not found in the data frame")
  }
  
  # Function to identify state
  identify_state <- function(text) {
    if (is.na(text)) {
      return(NA)  # Return NA if the input is NA
    }
    text <- as.character(text)
    # Check length and match appropriately
    if (nchar(text) == 2) {
      # Check for state abbreviation
      if (toupper(text) %in% state.abb) {
        return(toupper(text))
      }
    } else {
      # Check for full state name
      matched_state <- state.abb[match(toupper(text), toupper(state.name))]
      if (!is.na(matched_state)) {
        return(matched_state)
      }
      # If no full name found, check for abbreviation
      words <- unlist(strsplit(text, " "))
      for (word in words) {
        if (toupper(word) %in% state.abb) {
          return(toupper(word))
        }
      }
    }
    return(NA) # Return NA if no state found
  }
  
  # Apply the function to the specified column and create the new column
  df$identified_state <- sapply(df[[column_name]], identify_state)
  
  return(df)
}

############################################### Example Usage

# Set API Key for OpenAI
set_api_key <- function(api_key) {
  Sys.setenv(OPENAI_API_KEY = api_key)
}
# Replace with your actual API key
set_api_key("sk-YOUR-KEY")

# Read data
library(readr)
# Read data
imported_data <- read_csv("C:/Users/User/OneDrive/Desktop/ftc_noncompete/lp0-rkua-8mvh.csv")

# Filter out any missing a comment
imported_data <- imported_data %>%
  filter(!is.na(Comment))

# Apply function to clean state column
imported_data <- identify_states(imported_data, "State/Province")

# Filter to only those with valid state
imported_data <- imported_data %>%
  filter(!is.na(identified_state))

# Remove columns that only contain text which directs to seeing attachments
imported_data <- imported_data %>%
  filter(Comment != "See attached file(s)")

# Sample and process comments by state
processed_data <- process_and_sample(imported_data, 
                                     sample_size = 1250, 
                                     strata_column = "identified_state", 
                                     text_column = "Comment")

# Save data
write.csv(processed_data, "processed_data.csv", row.names = F, na = "")

# Get location file saved
getwd()

############################################### Validate

# Validation 1: Check that each of the 500 sampled comments has a stance
check_stance_assigned <- function(df) {
  return(all(!is.na(df$stance)) && nrow(df) == 1250)
}

# Validation 2: Verify that sampling covers all levels of the strata
check_strata_coverage <- function(df, strata_column) {
  sampled_strata <- unique(df[[strata_column]])
  original_strata <- unique(imported_data[[strata_column]])
  return(all(sampled_strata %in% original_strata))
}

# Validation 3: Confirm that each stance is a single numeric value
check_stance_format <- function(df) {
  stance_values <- as.integer(unlist(strsplit(df$stance, ";")))
  return(all(!is.na(stance_values)) && all(stance_values %in% 1:4))
}

# Perform validations
valid_stance_assigned <- check_stance_assigned(processed_data)
valid_strata_coverage <- check_strata_coverage(processed_data, "identified_state")
valid_stance_format <- check_stance_format(processed_data)

# Output results
cat("Stance Assigned to All Samples: ", valid_stance_assigned, "\n")
cat("All Strata Covered: ", valid_strata_coverage, "\n")
cat("Valid Stance Format: ", valid_stance_format, "\n")


############################################### Analysis

# Load necessary libraries
library(dplyr)
library(usmap)
library(ggplot2)

# Step 1: Aggregate county population for each state
state_population_2015 <- usmap::countypop %>%
  group_by(abbr) %>%
  summarise(total_population_2015 = sum(pop_2015))

# Calculate the national total population
national_population_2015 <- sum(state_population_2015$total_population_2015)

# Ensure 'processed_data' has valid state abbreviations
valid_states <- unique(usmap::countypop$abbr)
processed_data <- processed_data %>%
  filter(identified_state %in% valid_states)

# Step 2: Join with your data
joined_data <- processed_data %>%
  filter(!is.na(stance), stance %in% c(1, 2)) %>%
  group_by(identified_state) %>%
  summarise(
    in_favor = sum(stance == 1),
    against = sum(stance == 2)
  ) %>%
  left_join(state_population_2015, by = c("identified_state" = "abbr"))

# Step 3: Calculate net agreement normalized per 100,000 people
joined_data <- joined_data %>%
  mutate(
    in_favor_per_100k = (in_favor / total_population_2015) * 100000,
    against_per_100k = (against / total_population_2015) * 100000,
    net_agreement_per_100k = in_favor_per_100k - against_per_100k
  )

# Load US states map data
us_states_map <- usmap::us_map(regions = "states")

# Join your data with the map data
joined_map_data <- left_join(joined_data, us_states_map, by = c("identified_state" = "abbr"))

# Generate enhanced heatmap with net agreement per 100,000 people
ggplot(joined_map_data) +
  geom_polygon(aes(x = x, y = y, group = group, fill = net_agreement_per_100k)) +
  geom_polygon(aes(x = x, y = y, group = group), color = "black", fill = NA) + # Adding state borders
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0,
                       space = "Lab", name = "Net Agreement per 100k") +
  theme_minimal() + # Using a minimal theme for a cleaner look
  theme(legend.position = "right") + # Adjust legend position
  labs(title = "US Net Agreement", 
       subtitle = "Visualizing state-level agreement per 100,000 people",
       caption = "Data source: FTC rule to ban noncompetes") +
  coord_fixed(1.3) 


# Classify NAs as 'unknown' (4) and add labels
processed_data <- processed_data %>%
  mutate(stance = ifelse(is.na(stance), 4, stance),
         stance_label = case_when(
           stance == 1 ~ "In Favor",
           stance == 2 ~ "Against",
           stance == 3 ~ "Neutral",
           stance == 4 ~ "Unknown"
         ))

# Join with state population data
processed_data_with_pop <- processed_data %>%
  left_join(state_population_2015, by = c("identified_state" = "abbr"))

# Calculate overall percentages
overall_percentages <- processed_data_with_pop %>%
  count(stance_label, wt = total_population_2015) %>%
  mutate(percentage = n / sum(n) * 100)

# Create the Overall Bar Plot
ggplot(overall_percentages, aes(x = stance_label, y = percentage, fill = stance_label)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +
  labs(x = "Stance", y = "Percentage", title = "How do you feel about the FTC's proposed rule to ban noncompetes?", subtitle = "Overall Percentage by Stance") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal()

# Calculate state-wise percentages
state_percentages <- processed_data_with_pop %>%
  group_by(identified_state) %>%
  count(stance_label, wt = total_population_2015) %>%
  mutate(percentage = n / sum(n) * 100)

# Create trellis plot
ggplot(state_percentages, aes(x = stance_label, y = percentage, fill = stance_label)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5, position = position_dodge(width = 0.9), size = 2.5) +
  facet_wrap(~ identified_state) +
  labs(x = "Stance", y = "Percentage", title = "State-wise Feelings about the FTC's Rule on Noncompetes", subtitle = "Percentage by Stance in Each State") +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 100, 20)) + # y-axis labels in increments of 20%
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) # Rotate x-axis labels for readability

