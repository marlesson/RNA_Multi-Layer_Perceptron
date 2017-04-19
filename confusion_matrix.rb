# Implements Confusion Matrix for predictions values
#

class ConfusionMatrix

  # classes     = [classe1, classe2]
  # predictions = [[predict class, original class], [predict class, original class]]
  def initialize(classes, predictions)
    @classes     = classes
    @predictions = predictions
    @matrix      = []

    build_matrix
  end

  def to_s
    str = "\t\t"
    
    @classes.each do |c|
      str << "[#{c}]\t\t"
    end
    str << "\n"

    @classes.each do |c|
      str << "[#{c}]\t\t"      
      @classes.each do |c2|
        str << "#{@matrix[c][c2]}\t\t"  
      end
      str << "\n"
    end

    str
  end

  def count_correct_predictions
    @predictions.select{|p| p[0] == p[1]}.size.to_f
  end

  def predictions_per_class(klass)
    @predictions.select{|p| p[1] == klass}.size.to_f
  end

  def accuracy
    self.count_correct_predictions()/@predictions.size
  end

  def accuracy_per_class
    accuracy = {}

    @classes.each do |c|
      accuracy[c] = @matrix[c][c].to_f/predictions_per_class(c)
    end

    accuracy
  end

  def sensitivity

  end
  
  def specificity

  end

  def efficiency

  end

  protected 

  def build_matrix
    @classes.each do |c|
      @matrix[c] = []

      @classes.each do |c2|
        @matrix[c][c2] = 0

        @predictions.each do |pred| 
          pred_output, output  = pred
          @matrix[c][c2] += 1 if c == output and c2 == pred_output
        end
      end
    end
  end

end