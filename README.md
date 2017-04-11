# RNA_Multi-Layer_Perceptron

Artificial Neural Network Multi Layer Perceptron With Backpropagation. This RNA was based on the algorithm presented in https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.

## Use

### Training
```ruby
input  = [0.05, 0.1]
output = [0.01, 0.99] 

nn     = RNAMultiLayer.new(2, 2, [2, 2], 
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

```

### Prediction
```ruby
out = nn.output(input)
puts ">>> #{out}"
```

## Test

### run.rb
```ruby
ruby run.rb
```
