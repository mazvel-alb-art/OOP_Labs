#include "RandomForestRegressor.hpp"
#include "Exceptions.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <span>
#include <unordered_set>
#include <limits>
#include <thread>
#include <exception>

RandomForestRegressor::RandomForestRegressor(
    std::size_t n_estimators,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    MaxFeatures max_features,
    bool bootstrap,
    std::optional<std::size_t> max_samples,
    std::uint32_t random_state,
    std::size_t n_jobs
) : n_estimators_(n_estimators), max_depth_(max_depth), min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf), max_features_(max_features),
bootstrap_(bootstrap), max_samples_(max_samples), random_state_(random_state), n_jobs_(n_jobs){
    if (n_estimators_ < 1){
        throw std::invalid_argument("RandomForestRegressor: n_estimators < 1");
    }
    if (min_samples_split_ < 2){
        throw std::invalid_argument("RandomForestRegressor: min_samples_split < 2");
    }
    if (min_samples_leaf_ < 1){
        throw std::invalid_argument("RandomForestRegressor: min_samples_leaf < 1");
    }
    if (n_jobs_ < 1){
        throw std::invalid_argument("RandomForestRegressor: n_jobs < 1");
    }
    if (max_samples_.has_value() && max_samples_.value() == 0){
        throw std::invalid_argument("RandomForestRegressor: max_samples_.has_value() && max_samples_ == 0");
    }
    if (max_samples_.has_value() && bootstrap == false){
        throw std::invalid_argument("RandomForestRegressor: max_samples_.has_value() && bootstrap == false)");
    }
    if ((max_features.mode == MaxFeaturesMode::Count) && (max_features_.count == 0)){
        throw std::invalid_argument("RandomForestRegressor: (max_features.mode == MaxFeaturesMode::Count) && (max_features_.count == 0)");
    }
}

void RandomForestRegressor::fit(const Matrix& X, std::span<const double> y){
    if (X.rows() == 0) {
        throw EmptyTrainingSet("RandomForestRegressor::fit: empty training set");
    }
    if (X.rows() != y.size()) {
        throw DimensionMismatch("RandomForestRegressor::fit: X.rows() != y.size()");
    }
    if (X.cols() == 0) {
        throw DimensionMismatch("RandomForestRegressor::fit: X.cols() == 0");
    }
    if (max_samples_.has_value() && (max_samples_.value() > X.rows())){
        throw DimensionMismatch("RandomForestRegressor::fit: max_samples_.value() > X.rows()");
    }
    if ((max_features_.mode == MaxFeaturesMode::Count) && max_features_.count > X.cols()){
        throw DimensionMismatch("RandomForestRegressor::fit: max_features_.count > X.cols");
    }

    std::size_t n = X.rows();
    //seed
    std::mt19937 forest_rng(random_state_);

    std::vector<std::uint32_t> seeds;
    seeds.reserve(n_estimators_);

    std::unordered_set<std::uint32_t> used_seeds;

    if (static_cast<std::uint64_t>(n_estimators_) >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL) {
        throw std::invalid_argument("Too many trees for unique uint32_t seeds");
    }

    for (std::size_t i = 0; i < n_estimators_; ++i) {
        std::uint32_t seed;

        do {
            seed = forest_rng();
        } while (!used_seeds.insert(seed).second);

        seeds.push_back(seed);
    }


    //bootstrap
    std::size_t workers_count = std::min(n_jobs_, n_estimators_);

    std::vector<std::unique_ptr<CartRegressor>> new_trees(n_estimators_);

    std::vector<std::exception_ptr> errors(n_estimators_);

    {
        std::vector<std::jthread> workers;
        workers.reserve(workers_count);

        for (std::size_t worker_id = 0; worker_id < workers_count; ++worker_id) {

            workers.emplace_back([&, worker_id]() {

                for (std::size_t i = worker_id; i < n_estimators_; i += workers_count) {

                
                

                try{
                std::vector<std::size_t> counts(n, 0);

                std::mt19937 rng(seeds[i]);
                std::uniform_int_distribution<std::size_t> dist(0, n - 1);

                std::size_t m = max_samples_.value_or(n);

                if (bootstrap_ == true){
                    for (std::size_t j = 0; j < m;j++){
                        ++counts[dist(rng)];
                    }
                }
                else{
                    std::fill(counts.begin(), counts.end(), 1);
                }

                auto tree = std::make_unique<CartRegressor>(max_depth_, min_samples_split_, min_samples_leaf_, max_features_, seeds[i]);
                tree->fit_with_counts(X, y, counts); 
                new_trees[i] = std::move(tree);
                }
                catch (...) {
                errors[i] = std::current_exception();
                }
            }
                   
        
   
                
            });
        }
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

// Здесь все потоки завершились.
// Далее проверка исключений и сохранение ансамбля.
    trees_ = std::move(new_trees);
    num_features_ = X.cols();
    fitted_ = true;
}

std::vector<double> RandomForestRegressor::predict(const Matrix& X) const {
    if (!fitted_){
        throw NotFitted("RandomForestRegressor::predict: not fitted");
    }
    if (X.cols() != num_features_) {
        throw FeatureCountMismatch("RandomForestRegressor::predict: feature count mismatch");
    }
    std::vector<std::vector<double>> tree_predictions(n_estimators_);
    std::vector<double> result(X.rows(), 0);

    std::size_t workers_count = std::min(n_jobs_, n_estimators_);
    std::vector<std::exception_ptr> errors(n_estimators_);

{
    std::vector<std::jthread> workers;
    workers.reserve(workers_count);

    for (std::size_t worker_id = 0; worker_id < workers_count; ++worker_id) {

        workers.emplace_back([&, worker_id]() {

            for (std::size_t i = worker_id; i < n_estimators_; i += workers_count) {

                try {
                    tree_predictions[i] = trees_[i]->predict(X);
                }
                catch (...) {
                    errors[i] = std::current_exception();
                }
            }
        });
    }
}
for (const auto& error : errors) {
    if (error) {
        std::rethrow_exception(error);
    }
}
for (std::size_t i = 0;i < X.rows();i++){
    for (std::size_t j = 0; j < n_estimators_;j++){
        result[i] += tree_predictions[j][i];
    }
}
for (std::size_t i = 0;i < X.rows();i++){
    result[i] /= n_estimators_;
}
return result;
}