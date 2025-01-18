# frozen_string_literal: true

require 'faraday'
require 'json'
require 'logger'
require 'time'
require 'thread'  # For Mutex

module OpenAIChatUtility
  # ----------------------------------------------------------------------------
  # 1) Configuration & Custom Errors
  # ----------------------------------------------------------------------------

  # Thread-safe logging: You could use Rails.logger, semantic_logger, etc.
  # For demonstration, we create a basic STDOUT logger at INFO level by default.
  # In a Rails app, you'd replace `logger` references with `Rails.logger`.
  def self.logger
    @logger ||= Logger.new($stdout).tap do |log|
      log.level = Logger::INFO
    end
  end

  # Configuration for the OpenAI API, set via environment variables or a setter.
  # e.g.   ENV['OPENAI_API_BASE_URL'] = "https://api.openai.com"
  #        ENV['OPENAI_API_KEY'] = "sk-xxxxxxx"
  #
  # This is a trivial configuration store. For a large app, consider:
  # - Rails credentials
  # - dotenv / figaro for non-Rails
  # - Additional config options (timeouts, etc.)
  @config = {
    api_key:        ENV.fetch('OPENAI_API_KEY', nil),
    base_url:       ENV.fetch('OPENAI_API_BASE_URL', 'https://api.openai.com'),
    api_version:    ENV.fetch('OPENAI_API_VERSION', 'v1')
  }

  class << self
    attr_reader :config
  end

  # Set or override the API key programmatically.
  def self.set_api_key(key)
    @config[:api_key] = key
  end

  # Optionally set a custom base_url (staging, QA, etc.).
  def self.set_base_url(url)
    @config[:base_url] = url
  end

  # Custom error classes for structured error handling
  class RateLimitError < StandardError; end
  class BadRequestError < StandardError; end
  class APIError < StandardError; end
  class MissingSessionError < StandardError; end

  # ----------------------------------------------------------------------------
  # 2) Thread-Safe Session Store
  # ----------------------------------------------------------------------------

  # Use a Mutex to protect writes to the sessions hash in multithreaded environments.
  # If your app horizontally scales or is distributed, consider an external store (e.g. Redis).
  @sessions = {}
  @sessions_mutex = Mutex.new

  # Model token limits
  @model_token_limits = {
    'gpt-3.5-turbo' => 4096,
    'gpt-4'         => 8192,
    'gpt-4-32k'     => 32768
  }

  class << self
    attr_reader :model_token_limits
  end

  # ----------------------------------------------------------------------------
  # 3) Session Management
  # ----------------------------------------------------------------------------

  # Create a new session. Thread-safe.
  def self.create_session(
    session_id:,
    system_message: nil,
    model: 'gpt-3.5-turbo',
    max_context_tokens: 3000,
    overflow: 'none'
  )
    @sessions_mutex.synchronize do
      if @sessions.key?(session_id)
        raise APIError, "Session #{session_id} already exists. Clear or remove it first."
      end
      @sessions[session_id] = {
        system_message:      system_message,
        messages:            [],
        model:               model,
        max_context_tokens:  max_context_tokens,
        overflow:            overflow,
        log:                 []
      }
    end
  end

  # Clear session messages but keep system_message, logs, etc.
  def self.clear_session(session_id)
    @sessions_mutex.synchronize do
      session = @sessions[session_id]
      unless session
        logger.warn("Session #{session_id} does not exist. Nothing to clear.")
        return
      end
      session[:messages] = []
    end
  end

  # Remove session entirely
  def self.remove_session(session_id)
    @sessions_mutex.synchronize do
      unless @sessions.key?(session_id)
        logger.warn("Session #{session_id} does not exist. Nothing to remove.")
        return
      end
      @sessions.delete(session_id)
    end
  end

  # Reset system message
  def self.set_system_message(session_id, system_message)
    @sessions_mutex.synchronize do
      session = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist." unless session

      session[:system_message] = system_message
    end
  end

  # ----------------------------------------------------------------------------
  # 4) Utility to Append Messages
  # ----------------------------------------------------------------------------

  def self.append_user_message(session_id, content)
    @sessions_mutex.synchronize do
      session = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist." unless session

      session[:messages] << { role: 'user', content: content }
    end
  end

  def self.append_assistant_message(session_id, content)
    @sessions_mutex.synchronize do
      session = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist." unless session

      session[:messages] << { role: 'assistant', content: content }
    end
  end

  # ----------------------------------------------------------------------------
  # 5) Token Counting & Tools
  # ----------------------------------------------------------------------------

  def self.count_tokens(text)
    tokens = text.split(/(?<=\W)(?=\w)|(?<=\w)(?=\W)/)
    tokens.size
  end

  def self.get_session_token_count(session_id)
    @sessions_mutex.synchronize do
      s = @sessions[session_id]
      return 0 unless s

      all_text = ''
      all_text << "#{s[:system_message]}\n" if s[:system_message]
      s[:messages].each { |m| all_text << "#{m[:content]}\n" }
      count_tokens(all_text)
    end
  end

  def self.get_model_token_limit(model)
    model_token_limits[model] || 4096
  end

  # ----------------------------------------------------------------------------
  # 6) Summaries & Chunked Summaries
  # ----------------------------------------------------------------------------

  def self.chunk_text_by_token_count(text, chunk_size)
    all_tokens = text.split(/(?<=\W)(?=\w)|(?<=\w)(?=\W)/)
    chunks = []
    buffer = []
    count  = 0

    all_tokens.each do |tk|
      if (count + 1) > chunk_size
        chunks << buffer.join
        buffer = [tk]
        count  = 1
      else
        buffer << tk
        count += 1
      end
    end
    chunks << buffer.join unless buffer.empty?
    chunks
  end

  def self.chunked_summarize_text(
    text:,
    chunk_size: 2000,
    summarization_model: 'gpt-3.5-turbo',
    temperature: 0.2,
    max_tokens: 200
  )
    token_count = count_tokens(text)
    if token_count <= chunk_size
      return gpt_api(
        prompt: "Please summarize:\n\n#{text}",
        model:  summarization_model,
        temperature: temperature,
        max_tokens:  max_tokens
      )
    end

    text_chunks = chunk_text_by_token_count(text, chunk_size)
    if text_chunks.size == 1
      return gpt_api(
        prompt: "Please summarize:\n\n#{text_chunks.first}",
        model: summarization_model,
        temperature: temperature,
        max_tokens: max_tokens
      )
    end

    partial_summaries = text_chunks.map.with_index do |chunk, idx|
      gpt_api(
        prompt: "Please summarize chunk #{idx+1}:\n\n#{chunk}",
        model: summarization_model,
        temperature: temperature,
        max_tokens: max_tokens
      )
    end

    combined_prompt = <<~TXT
      Please combine and refine these chunk summaries:

      #{partial_summaries.map.with_index { |s,i| "Chunk #{i+1} Summary:\n#{s}" }.join("\n\n")}
    TXT

    final_summary = gpt_api(
      prompt: combined_prompt,
      model: summarization_model,
      temperature: temperature,
      max_tokens: max_tokens
    )
    final_summary
  end

  def self.summarize_session(
    session_id:,
    summarization_model: 'gpt-3.5-turbo',
    prompt_prefix: "Please summarize the following conversation in a concise way while preserving important context:",
    chunked: false,
    chunk_size: 2000
  )
    @sessions_mutex.synchronize do
      s = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist." unless s

      conversation_text = ''
      conversation_text << "System: #{s[:system_message]}\n" if s[:system_message]

      s[:messages].each do |m|
        role = m[:role].capitalize
        conversation_text << "#{role}: #{m[:content]}\n"
      end

      total_tokens = count_tokens(conversation_text)
      do_chunked = chunked || (total_tokens > 4 * chunk_size)

      summary = if do_chunked
        chunked_summarize_text(
          text: conversation_text,
          chunk_size: chunk_size,
          summarization_model: summarization_model,
          temperature: 0.2,
          max_tokens: 200
        )
      else
        prompt_to_summarize = "#{prompt_prefix}\n\n#{conversation_text}"
        begin
          gpt_api(
            prompt: prompt_to_summarize,
            model: summarization_model,
            temperature: 0.2,
            max_tokens: 200
          )
        rescue => e
          logger.error("Summarization call failed: #{e.message}")
          '**[Unable to Summarize - Too Large or Error Occurred]**'
        end
      end

      s[:system_message] = "Summary of conversation:\n#{summary}"
      s[:messages] = []
    end
  end

  def self.truncate_session(session_id)
    @sessions_mutex.synchronize do
      s = @sessions[session_id]
      return unless s

      loop do
        total_tokens = get_session_token_count(session_id)
        break if total_tokens <= s[:max_context_tokens]
        break if s[:messages].empty?

        s[:messages].shift
      end
    end
  end

  # ----------------------------------------------------------------------------
  # 7) Logging and Usage Tracking
  # ----------------------------------------------------------------------------

  def self.add_log_record(session_id:, user_prompt:, assistant_response:, usage_info:)
    @sessions_mutex.synchronize do
      @sessions[session_id][:log] << {
        timestamp: Time.now.iso8601,
        user_prompt: user_prompt,
        assistant_response: assistant_response,
        usage: usage_info
      }
    end
  end

  def self.get_session_log(session_id)
    @sessions_mutex.synchronize do
      session = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist." unless session

      session[:log]
    end
  end

  # ----------------------------------------------------------------------------
  # 8) High-level Chat Method
  # ----------------------------------------------------------------------------

  def self.gpt_chat(
    session_id:,
    user_prompt:,
    use_context: true,
    model: nil,
    temperature: 0.5,
    max_tokens: 50,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    verbose: false,
    num_retries: 3,
    pause_base: 1,
    summarization_model: 'gpt-3.5-turbo'
  )
    # Use a lock to safely read/write session data
    session_data = @sessions_mutex.synchronize do
      s = @sessions[session_id]
      raise MissingSessionError, "Session #{session_id} does not exist. Create one first." unless s
      s
    end

    model ||= session_data[:model]

    unless use_context
      # Single-turn usage
      response_text = gpt_api(
        prompt: user_prompt,
        model: model,
        temperature: temperature,
        max_tokens: max_tokens,
        presence_penalty: presence_penalty,
        frequency_penalty: frequency_penalty,
        verbose: verbose,
        num_retries: num_retries,
        pause_base: pause_base
      )
      return response_text
    end

    # If using context:
    append_user_message(session_id, user_prompt)
    total_tokens = get_session_token_count(session_id)
    model_limit  = get_model_token_limit(model)

    if (total_tokens + max_tokens) > model_limit
      logger.warn("Requested tokens (#{max_tokens}) + conversation size (#{total_tokens}) may exceed model limit (#{model_limit}).")
    end

    if total_tokens > session_data[:max_context_tokens]
      case session_data[:overflow]
      when 'summarize'
        summarize_session(session_id: session_id, summarization_model: summarization_model, chunked: true)
      when 'truncate'
        truncate_session(session_id)
      when 'none'
        logger.warn("Context token limit exceeded, overflow strategy is 'none'. Request may fail.")
      end
    end

    if get_session_token_count(session_id) > session_data[:max_context_tokens]
      logger.warn("Even after summarization/truncation, token count is still too high.")
    end

    # Build final messages
    final_messages = []
    @sessions_mutex.synchronize do
      final_messages << { role: 'system', content: session_data[:system_message] } if session_data[:system_message]
      final_messages.concat(session_data[:messages])
    end

    response_text = gpt_api(
      prompt: nil,
      model: model,
      temperature: temperature,
      max_tokens: max_tokens,
      messages_list: final_messages,
      presence_penalty: presence_penalty,
      frequency_penalty: frequency_penalty,
      verbose: verbose,
      num_retries: num_retries,
      pause_base: pause_base
    )

    append_assistant_message(session_id, response_text)

    usage_info = response_text.instance_variable_get(:@usage_info)
    if usage_info
      add_log_record(
        session_id: session_id,
        user_prompt: user_prompt,
        assistant_response: response_text,
        usage_info: usage_info
      )
    end

    response_text
  end

  # ----------------------------------------------------------------------------
  # 9) Low-Level gpt_api Function with Faraday & Exponential Backoff
  # ----------------------------------------------------------------------------

  def self.gpt_api(
    prompt: nil,
    model: 'gpt-3.5-turbo',
    temperature: 0.5,
    max_tokens: 50,
    messages_list: nil,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    verbose: false,
    num_retries: 3,
    pause_base: 1
  )
    raise APIError, 'OpenAI API key is missing. Set it using set_api_key().' if config[:api_key].to_s.empty?

    # Construct payload
    messages = if messages_list
                 messages_list
               else
                 raise APIError, 'You must provide either prompt or messages_list.' unless prompt
                 [{ role: 'user', content: prompt }]
               end

    body = {
      model: model,
      temperature: temperature,
      max_tokens: max_tokens,
      messages: messages,
      presence_penalty: presence_penalty,
      frequency_penalty: frequency_penalty
    }

    logger.debug("API Request Body: #{JSON.pretty_generate(body)}") if verbose

    # Build Faraday connection
    conn = Faraday.new(url: "#{config[:base_url]}/#{config[:api_version]}") do |faraday|
      faraday.request  :json
      faraday.response :json, content_type: /\bjson$/
      faraday.adapter  Faraday.default_adapter
    end

    attempt = 0
    response = nil
    begin
      attempt += 1
      response = conn.post('chat/completions') do |req|
        req.headers['Authorization'] = "Bearer #{config[:api_key]}"
        req.headers['Content-Type']  = 'application/json'
        req.body = body
      end

      # Check HTTP status
      unless response.success?
        parse_and_raise_error(response)
      end
    rescue => e
      # Exponential backoff
      if attempt < num_retries
        delay = pause_base * (2 ** (attempt - 1))
        logger.warn("Request failed (attempt #{attempt}): #{e.message}. Retrying in #{delay} seconds.")
        sleep(delay)
        retry
      else
        logger.error("Request failed after #{attempt} attempts: #{e.message}")
        raise APIError, e.message
      end
    end

    logger.debug("Raw Response: #{response.body}") if verbose

    parsed = response.body
    if parsed['choices']&.any?
      message_content = parsed['choices'].first.dig('message', 'content') || ''
    else
      message_content = 'The model did not return a message. Possibly increase max_tokens or check prompt.'
    end

    usage_info = parsed['usage'] # e.g., { 'prompt_tokens'=>..., 'completion_tokens'=>..., ... }
    clean_message = message_content.gsub(/\n+/, "\n").strip
    clean_message.instance_variable_set(:@usage_info, usage_info) if usage_info

    clean_message
  end

  def self.parse_and_raise_error(response)
    status = response.status.to_i
    body   = response.body || {}
    err_msg = body.dig('error', 'message') || response.reason_phrase

    case status
    when 429
      raise RateLimitError, "Rate limit error (429): #{err_msg}"
    when 400
      raise BadRequestError, "Bad request (400): #{err_msg}"
    else
      raise APIError, "HTTP error #{status} - #{err_msg}"
    end
  end

  # ----------------------------------------------------------------------------
  # 10) Cost Estimation & Embeddings
  # ----------------------------------------------------------------------------

  def self.estimate_cost(input_tokens:, output_tokens:, model: 'gpt-3.5-turbo')
    if model == 'gpt-4'
      price_1k_in  = (input_tokens  <= 8000) ? 0.03 : 0.06
      price_1k_out = (output_tokens <= 8000) ? 0.06 : 0.12
    else
      price_1k_in  = 0.002
      price_1k_out = 0.002
    end

    input_cost  = (input_tokens  / 1000.0) * price_1k_in
    output_cost = (output_tokens / 1000.0) * price_1k_out
    total_cost  = input_cost + output_cost

    {
      input_tokens:  input_tokens,
      output_tokens: output_tokens,
      input_cost:    input_cost,
      output_cost:   output_cost,
      total_cost:    total_cost
    }
  end

  # text_to_embeddings using Faraday
  def self.text_to_embeddings(
    text:,
    model: 'text-embedding-ada-002',
    num_retries: 3,
    pause_base: 1
  )
    raise APIError, 'OpenAI API key is missing. Set it using set_api_key().' if config[:api_key].to_s.empty?

    inputs = text.is_a?(Array) ? text : [text]
    body = { model: model, input: inputs }

    conn = Faraday.new(url: "#{config[:base_url]}/#{config[:api_version]}") do |faraday|
      faraday.request  :json
      faraday.response :json
      faraday.adapter  Faraday.default_adapter
    end

    attempt = 0
    response = nil

    begin
      attempt += 1
      response = conn.post('embeddings') do |req|
        req.headers['Authorization'] = "Bearer #{config[:api_key]}"
        req.headers['Content-Type']  = 'application/json'
        req.body = body
      end

      unless response.success?
        parse_and_raise_error(response)
      end
    rescue => e
      if attempt < num_retries
        delay = pause_base * (2 ** (attempt - 1))
        logger.warn("Embeddings request failed (attempt #{attempt}): #{e.message}. Retrying in #{delay}s.")
        sleep(delay)
        retry
      else
        logger.error("Embeddings request failed after #{attempt} attempts: #{e.message}")
        raise APIError, e.message
      end
    end

    parsed = response.body
    data   = parsed['data']
    raise APIError, 'The API did not return any embeddings.' unless data

    all_embeddings = data.map { |item| item['embedding'] }
    return all_embeddings.first if all_embeddings.size == 1
    all_embeddings
  end
end

# ----------------------------------------------------------------------------
# EXAMPLE USAGE (comment out or remove in production):
# ----------------------------------------------------------------------------
if __FILE__ == $PROGRAM_NAME
  OpenAIChatUtility.logger.level = Logger::DEBUG  # For demonstration; set to INFO/ERROR in production

  # Set or override config
  # OpenAIChatUtility.set_api_key("sk-XXXXXX")
  # OpenAIChatUtility.set_base_url("https://api.openai.com")  # or a staging endpoint, etc.

  # Create a session and test
  OpenAIChatUtility.create_session(session_id: "mychat", system_message: "You are a helpful assistant.")
  resp1 = OpenAIChatUtility.gpt_chat(session_id: "mychat", user_prompt: "Hello, how are you?")
  puts "Assistant: #{resp1}"

  resp2 = OpenAIChatUtility.gpt_chat(session_id: "mychat", user_prompt: "What is the capital of France?")
  puts "Assistant: #{resp2}"

  logs = OpenAIChatUtility.get_session_log("mychat")
  p logs

  # Clean up
  OpenAIChatUtility.clear_session("mychat")
  OpenAIChatUtility.remove_session("mychat")
end
