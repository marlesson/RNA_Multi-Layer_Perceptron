require_relative 'neuron'

class NeuronLayer
  
  attr_reader :neurons

  def initialize(neurons = [])
    @neurons = neurons
  end

  def outputs
    @neurons.collect{|n| n.output}
  end

  def feed_forward(inputs)
    @neurons.collect{|n| n.output(inputs)}
  end

  def to_s
    @neurons.collect(&:to_s).to_sentence
  end
end