#include "KnnClassifier.hpp"
#include "Exceptions.hpp"
#include <map>

KnnClassifier::KnnClassifier(std::unique_ptr<IDistance> metric, std::size_t k, Weighting weighting)
    : KnnBase(std::move(metric), k, weighting)
{
}

std::vector<double> KnnClassifier::predict(const Matrix& features) const {
    if (train_features_ == nullptr){
        throw NotFitted("KnnClassifier::predict: модель не обучена");
    }
    if (features.cols() != train_features_->cols()){
        throw FeatureCountMismatch("KnnClassifier::predict: число признаков не совпало");
    }

    std::vector<double> result;
    result.reserve(features.rows());

    for (std::size_t i = 0; i < features.rows(); i++){
        std::span<const double> row = features[i];
        auto neighbors = find_neighbors(row, k_);

        if (weighting_ == Weighting::Uniform){
            std::map<double, std::size_t> votes;
            for (const auto& n : neighbors){
                votes[train_target_[n.second]]++;
            }

            double best_class = 0.0;
            std::size_t best_votes = 0;
            for (const auto& [cls, cnt] : votes){
                if (cnt > best_votes){
                    best_votes = cnt;
                    best_class = cls;
                }
            }
            result.push_back(best_class);
        }
        else {   // Weighting::Distance
            bool exact_match = false;
            double exact_class = 0.0;
            for (const auto& n : neighbors){
                if (n.first == 0.0){
                    exact_match = true;
                    exact_class = train_target_[n.second];
                    break;
                }
            }

            if (exact_match){
                result.push_back(exact_class);
                continue;
            }

            std::map<double, double> weighted_votes;
            for (const auto& n : neighbors){
                double w = 1.0 / n.first;
                weighted_votes[train_target_[n.second]] += w;
            }

            double best_class = 0.0;
            double best_weight = 0.0;
            for (const auto& [cls, w] : weighted_votes){
                if (w > best_weight){
                    best_weight = w;
                    best_class = cls;
                }
            }
            result.push_back(best_class);
        }
    }

    return result;
}