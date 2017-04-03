#!/usr/bin/env ruby

require_relative 'ml_perceptron'


input  = [[0.05, 0.1]]
output = [[0.01, 0.99]] 

nn  = MLPerceptron.new
out = nn.train(input, output)

out = nn.run(input[0])

puts "#{nn.inspect}"
puts ">>> #{out}"

#nn.train([0.05, 0.1], [0.01, 0.99])


