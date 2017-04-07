# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

require_relative 'neuron_layer'

class RNAMultiLayer

  def initialize(num_inputs, num_outputs, num_neuros_per_layer = [], weights_per_neuron = [])
    # Parameters RNA
    @learning_rate = 0.5

    @num_inputs    = num_inputs
    @num_outputs   = num_outputs
    @output        = []

    @layers        = []

    build_layers_connections(num_neuros_per_layer, weights_per_neuron)
  end

  def output(inputs)
    
    @output = @layers.collect do |l|
      inputs = l.feed_forward(inputs)
    end.last
  end

  def last_output
    @output
  end

  def train(inputs, outputs)
    output(inputs)

    # 1. Output Neuron Deltas
    pd_errors_wrt_output_neuron_total_net_input = {}
    @output_layer.neurons.each_index do |n|
      # ∂E/∂zⱼ
      pd_errors_wrt_output_neuron_total_net_input[n] = @output_layer.neurons[n].calculate_pd_error_wrt_total_net_input(outputs[n])
      
    end

    # 2. Hidden Neuron Deltas
    pd_errors_wrt_hidden_neuron_total_net_input = {}
    @hidden_layer.neurons.each_index do |h|

      # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
      # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
      d_error_wrt_hidden_neuron_output = 0
      @output_layer.neurons.each_index do |o|
        d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * @output_layer.neurons[o].weights[h]
      end
      # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
      pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * @hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()    
    end


    # 3. Update output neuron weights
    @output_layer.neurons.each_index do |o|
      @output_layer.neurons[o].weights.each_index do |w_ho|
    
        # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
        pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * @output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
        # Δw = α * ∂Eⱼ/∂wᵢ
        @output_layer.neurons[o].weights[w_ho] += (@learning_rate * pd_error_wrt_weight) * (-1)
      end

      # Update BIAS
      # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
      pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] *  1     
      # Δw = α * ∂Eⱼ/∂wᵢ
      @output_layer.neurons[o].bias += (@learning_rate * pd_error_wrt_weight) * (-1)      
    end


    # 4. Update hidden neuron weights
    @hidden_layer.neurons.each_index do |h|
      @hidden_layer.neurons[h].weights.each_index do |w_ih|

        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * @hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

        # Δw = α * ∂Eⱼ/∂wᵢ
        @hidden_layer.neurons[h].weights[w_ih] += @learning_rate * pd_error_wrt_weight * (-1)
      end

      # Update BIAS
      # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
      pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] *  1     
      # Δw = α * ∂Eⱼ/∂wᵢ
      @hidden_layer.neurons[h].bias += (@learning_rate * pd_error_wrt_weight) * (-1)          
    end
  end

  def calculate_total_error(training_sets)
    total_error = 0
    
    training_sets.each_index do |t|
      training_inputs, training_outputs = training_sets[t]
      output(training_inputs)
      
      training_outputs.each_index do |o|
        total_error += @output_layer.neurons[o].calculate_error(training_outputs[o])
      end
    end

    total_error
  end

  def to_s
    str = "RNA: \n"

    @layers.each_index do |i| 
      str << "Layer #{i}: \n"
      str << @layers[i].to_s
      str << "\n\n"
    end

    str
  end

  private

  def build_layers_connections(num_neuros_per_layer, weights_per_neuron)

    # Hidden Layer
    neurons = []
    num_neuros_per_layer[0].times do |h_layer|
      neurons << Neuron.new(weights_per_neuron[0][h_layer][1..-1], weights_per_neuron[0][h_layer][0])
    end
    @hidden_layer = NeuronLayer.new(neurons)

    
    # Output Layer
    neurons = []
    num_neuros_per_layer[1].times do |o_layer|
      neurons << Neuron.new(weights_per_neuron[1][o_layer][1..-1], weights_per_neuron[1][o_layer][0])
    end    
    @output_layer = NeuronLayer.new(neurons)      

    @layers = [@hidden_layer, @output_layer]
  end
end