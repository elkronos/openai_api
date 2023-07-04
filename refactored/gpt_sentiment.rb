require 'net/http'
require 'json'

def gpt_sentiment(df, system_message = "You are an expert sentiment analyzer, analyzing emotion from text. Please carefully read the following text and respond with '-1' if the sentiment is negative, '0' if it's neutral, or '1' if it's positive.", temperature = 0.2)
  sentiment_list = []
  flags = []

  df.each do |row|
    text = row['Text']

    # Check if the text input is not empty
    if text.length > 0
      sentiment = gpt_api(text, model: 'gpt-3.5-turbo', temperature: temperature, max_tokens: 2000, num_retries: 3, pause_base: 1, presence_penalty: 0.0, frequency_penalty: 0.0, system_message: system_message)

      # Filter the sentiment value
      sentiment_numbers = sentiment.scan(/-1|0|1/).map(&:to_i)

      if sentiment_numbers.empty?
        flags << "Invalid number"
        sentiment_list << nil
      else
        sentiment_list << sentiment_numbers
        flags << (sentiment_numbers.length > 1 ? "Multiple valid numbers" : "Single valid number")
      end
    else
      flags << "Empty text"
      sentiment_list << nil
    end
  end

  df.map.with_index { |row, i| { ID: row['ID'], sentiment: sentiment_list[i], flag: flags[i] } }
end
