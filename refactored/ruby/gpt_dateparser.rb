require 'faraday'
require 'dotenv/load'
require 'date'
require 'json'

def set_api_key(api_key)
  ENV['OPENAI_API_KEY'] = api_key
end

set_api_key("sk-your-api-key")

def gpt_api(prompt, model = "gpt-3.5-turbo", temperature = 0.5, max_tokens = 50, system_message = nil, num_retries = 3, pause_base = 1, presence_penalty = 0.0, frequency_penalty = 0.0)
  messages = [{role: "user", content: prompt}]
  
  messages.prepend({role: "system", content: system_message}) unless system_message.nil?

  body = {
    model: model,
    temperature: temperature,
    max_tokens: max_tokens,
    messages: messages,
    presence_penalty: presence_penalty,
    frequency_penalty: frequency_penalty
  }

  response = nil
  num_retries.times do |i|
    response = Faraday.post("https://api.openai.com/v1/chat/completions", body.to_json, {"Authorization" => "Bearer #{ENV['OPENAI_API_KEY']}", "Content-Type" => "application/json"})
    break if response.status == 200
    sleep(pause_base)
  end

  if response.status != 200
    puts "The API request failed with status #{response.status}"
    return nil
  end

  content = JSON.parse(response.body)
  if content['choices'].length > 0
    message = content['choices'][0]['message']['content']
  else
    message = "The model did not return a message. You may need to increase max_tokens."
  end

  clean_message = message.gsub("\n", " ").strip # replace newlines with spaces and trim white spaces
  return clean_message
end

def gpt_date_parser(df, column, system_message = "You are an expert date identifier. You will be given text which you must read and determine if it's a real date. If it is a real date, restate it in YYYY-MM-DD format, surrounded with '#'. If there is no date, state that there is no date.", temperature = 0.1)
  parsed_df = []
  failed_df = []

  df.each_with_index do |row, i|
    id = i
    date = row[column]
    begin
      parsed_date = Date.parse(date)
      parsed_df << {id: id, original: date, parsed: parsed_date.to_s, status: 'parsed', parsed_by: 'Ruby'}
      puts "Parsed: #{date}"
    rescue
      failed_df << {id: id, original: date, parsed: nil, status: 'failed', parsed_by: 'NA'}
      puts "Failed to parse: #{date}"
    end
  end

  failed_df.each_with_index do |row, i|
    original_date = row[:original]
    prompt = "What date is specified in '#{original_date}'?"

    response = gpt_api(prompt, system_message: system_message, temperature: temperature)
    response_date = response[/#.*#/]

    unless response_date.nil?
      response_date = response_date.gsub("#", "").strip

      begin
        final_date = Date.parse(response_date)
        row[:parsed_by] = 'GPT'
        row[:parsed] = final_date.to_s
      rescue
        row[:parsed_by] = 'NA'
        row[:parsed] = nil
      end
    else
      row[:parsed_by] = 'NA'
      row[:parsed] = nil
    end

    puts "Response from GPT: #{response_date}"
  end

  result_df = parsed_df + failed_df
  result_df.sort_by! { |row| row[:id] }
  return result_df
end