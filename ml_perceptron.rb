class MLPerceptron

  OUTPUT  = 1

  def initialize(opts = {})
    @opts = opts

    @opts[:max_iterations] ||= 1_000
    @opts[:max_error]      ||= 0.01
    @opts[:log_every]      ||= 100

    set_initial_values
  end

  def train(inputs, expected_outputs)
    iteration, error = 0, 0

    set_initial_values

    while iteration < @opts[:max_iterations]
      iteration += 1

      error     = train_on_batch(inputs, expected_outputs)
      
      if (iteration % @opts[:log_every] == 0)
        puts "[#{iteration}] #{(error * 100).round(2)}% mse"
      end

      break if @opts[:max_error] && (error < @opts[:max_error])
    end

    { 
      error: error.round(5), 
      iterations: iteration, 
      below_error_threshold: (error < @opts[:max_error])
    }
  end

  def run(inputs)
    #Layer 1, last out is input
    last_out = inputs

    @weights.each_index do |layer|

      last_out.each_index do |i|
        _net   = net(layer, i, last_out.dup)
        g_net  = sigmoid(_net)

        @outputs[layer][i] = g_net
      end

      #Layer 2, last out is last out
      last_out = @outputs[layer]
    end

    last_out
  end

  private

  def set_initial_values

    #set default weigths
    @weights = [
      [
        [0.15, 0.25, 0.35], #W1, W3, B1
        [0.2,  0.3,  0.35]  #W2, W4, B2
      ],
      [
        [0.4,  0.5,  0.6], #W5, W6, B3
        [0.45, 0.55, 0.6]  #W6, W8, B4
      ] 
    ]

    #set default outputs
    @outputs = [
      [0, 0], #g(h1), g(h2)
      [0, 0]  #g(o1), g(o2)
    ]  
      
  end

  def net(layer, neural, inputs)
    net    = 0
    inputs << 1 #Bias
    
    inputs.each_index do |i|
      input = inputs[i]
      net   += @weights[layer][neural][i]*input
    end
    
    net
  end

  def sigmoid(x)
    1 / (1 + Math.exp(-x))
  end

  def train_on_batch(inputs, expected_outputs)
    total_mse = 0

    inputs.each.with_index do |input, i|
      self.run(input)

      training_error = calculate_training_error(expected_outputs[i])
      total_mse     += mean_squared_error(training_error)

      update_gradients(training_error)
    end

    total_mse / inputs.length.to_f # average mean squared error for batch
  end

  def calculate_training_error(expected_outputs)
    @outputs[-1].map.with_index do |output, i| 
      output - expected_outputs[i]
    end
  end

  def mean_squared_error(errors)
    errors.map {|e| e**2}.reduce(:+) / errors.length.to_f
  end  

  def update_gradients(errors)

  end
end