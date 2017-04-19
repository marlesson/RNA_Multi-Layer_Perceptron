# Implements 'one hot encode' for inputs

class OneHotEncode

  # data = [
  #         [feature, feature, feature...],
  #         [feature, feature, feature...],
  #        ]
  def initialize(data)
    @data           = data
    @encode_feature = {}

    build_hash
  end

  def count_features
    @data.first.size
  end

  # features = [feature, feature, feature]
  def encode_features(features)
    enc = []

    features.size.times.each do |i|
      if @encode_feature[i]
        enc << @encode_feature[i][features[i]]
      else
        enc << features[i]
      end
    end
    enc
  end

  private

  def build_hash
    features_data = {}

    # change structs
    @data.each do |data|
      count_features.times do |i|
        features_data[i] ||= []
        features_data[i] << data[i]
      end
    end

    # Uniq features
    count_features.times do |i|
      features_data[i].uniq!
    end

    # encode features
    encode_feature = {}
    count_features.times do |i|
      encode_feature[i] = {}

      features_data[i].each do |v|
        encode_feature[i][v] = encode(v, features_data[i])
      end
    end

    @encode_feature = encode_feature
  end

  def encode(value, values)
    values_enc = [0]*values.size
    values_enc[values.index(value)] = 1
    values_enc
  end
end