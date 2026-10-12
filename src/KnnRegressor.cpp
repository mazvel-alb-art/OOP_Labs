#include "KnnRegressor.hpp"
#include "Exceptions.hpp"



KnnRegressor::KnnRegressor(std::unique_ptr<IDistance> metric, std::size_t k, Weighting weighting) : KnnBase(std::move(metric), k, weighting)
{
}
//
std::vector<double> KnnRegressor::predict(const Matrix& features) const {
    if (train_features_ == nullptr){
        throw NotFitted("KnnRegressor::predict: модель не обучена");
    }
    if (features.cols() != train_features_->cols()){
        throw FeatureCountMismatch("KnnRegressor::predict: число признаков не совпало");
    }

    std::vector<double> result;
    result.reserve(features.rows());

    for (std::size_t i = 0; i < features.rows(); i++){
        std::span<const double> row = features[i];
        auto neighbors = find_neighbors(row, k_);

        if (weighting_ == Weighting::Uniform){
            double sum = 0.0;
            for (const auto& n : neighbors){
                sum += train_target_[n.second];
            }
            result.push_back(sum / static_cast<double>(neighbors.size()));
        }
        else {   // Weighting::Distance
            bool exact_match = false;
            double exact_target = 0.0;
            for (const auto& n : neighbors){
                if (n.first == 0.0){
                    exact_match = true;
                    exact_target = train_target_[n.second];
                    break;
                }
            }

            if (exact_match){
                result.push_back(exact_target);
                continue;
            }

            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            for (const auto& n : neighbors){
                double w = 1.0 / n.first;
                weighted_sum += w * train_target_[n.second];
                weight_sum += w;
            }
            result.push_back(weighted_sum / weight_sum);
        }
    }

    return result;
}