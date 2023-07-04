require 'net/http'
require 'json'

def gpt_api(prompt, model: 'gpt-3.5-turbo', temperature: 0.5, max_tokens: 50,
            system_message: nil, num_retries: 3, pause_base: 1,
            presence_penalty: 0.0, frequency_penalty: 0.0)

  messages = [{ 'role' => 'user', 'content' => prompt }]
  messages.unshift({ 'role' => 'system', 'content' => system_message }) if system_message

  url = URI.parse('https://api.openai.com/v1/chat/completions')
  headers = { 'Content-Type' => 'application/json',
              'Authorization' => "Bearer #{ENV['OPENAI_API_KEY']}" }

  response = nil
  num_retries.times do |retry_count|
    http = Net::HTTP.new(url.host, url.port)
    http.use_ssl = true

    request = Net::HTTP::Post.new(url.path, headers)
    request.body = {
      'model' => model.to_s,
      'temperature' => temperature,
      'max_tokens' => max_tokens,
      'messages' => messages,
      'presence_penalty' => presence_penalty,
      'frequency_penalty' => frequency_penalty
    }.to_json

    response = http.request(request)
    break if response.is_a?(Net::HTTPSuccess)
    sleep pause_base * (retry_count + 1)
  end

  unless response.is_a?(Net::HTTPSuccess)
    puts "Failed to make a successful API request. Error: #{response.body}"
    return nil
  end

  choices = JSON.parse(response.body)['choices']
  if choices.length > 0
    message = choices[0]['message']['content']
  else
    message = 'The model did not return a message. You may need to increase max_tokens.'
  end

  clean_message = message.gsub("\n", ' ') # replace newlines with spaces
  clean_message.strip! # trim leading and trailing white spaces
  clean_message
end