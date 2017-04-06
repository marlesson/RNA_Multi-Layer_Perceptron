#!/usr/bin/env ruby

require_relative 'rna_multi_layer'

input  = [0.05, 0.1]
output = [0.01, 0.99] 

nn  = RNAMultiLayer.new(2, 2, [2, 2], 
        [
          [
            [0.35, 0.15, 0.2],
            [0.35, 0.25, 0.3]
          ],
          [
            [0.6, 0.4, 0.45],
            [0.6, 0.5, 0.55]
          ]
        ])

10000.times.each do |t|
  puts "#{t} > #{nn.calculate_total_error([[input, output]])}"
  nn.train(input, output)
end

out = nn.output(input)

puts "#{nn.inspect}"
puts ">>> #{out}"



# [3300] 0.01% mse
# [3400] 0.01% mse
# {:error=>0.0001, :iterations=>3498, :below_error_threshold=>true}
# >>> [0.021770121359572105, 0.9821664561748541]


