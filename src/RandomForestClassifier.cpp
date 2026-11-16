#include "RandomForestClassifier.hpp"
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
#include <cmath>

RandomForestClassifier::RandomForestClassifier(
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
        throw std::invalid_argument("RandomForestClassifier: n_estimators < 1");
    }
    if (min_samples_split_ < 2){
        throw std::invalid_argument("RandomForestClassifier: min_samples_split < 2");
    }
    if (min_samples_leaf_ < 1){
        throw std::invalid_argument("RandomForestClassifier: min_samples_leaf < 1");
    }
    if (n_jobs_ < 1){
        throw std::invalid_argument("RandomForestClassifier: n_jobs < 1");
    }
    if (max_samples_.has_value() && max_samples_.value() == 0){
        throw std::invalid_argument("RandomForestClassifier: max_samples_.has_value() && max_samples_ == 0");
    }
    if (max_samples_.has_value() && bootstrap == false){
        throw std::invalid_argument("RandomForestClassifier: max_samples_.has_value() && bootstrap == false)");
    }
    if ((max_features.mode == MaxFeaturesMode::Count) && (max_features_.count == 0)){
        throw std::invalid_argument("RandomForestClassifier: (max_features.mode == MaxFeaturesMode::Count) && (max_features_.count == 0)");
    }
}

void RandomForestClassifier::fit(const Matrix& X, std::span<const double> y){
    if (X.rows() == 0) {
        throw EmptyTrainingSet("RandomForestClassifier::fit: empty training set");
    }
    if (X.rows() != y.size()) {
        throw DimensionMismatch("RandomForestClassifier::fit: X.rows() != y.size()");
    }
    if (X.cols() == 0) {
        throw DimensionMismatch("RandomForestClassifier::fit: X.cols() == 0");
    }
    if (max_samples_.has_value() && (max_samples_.value() > X.rows())){
        throw DimensionMismatch("RandomForestClassifier::fit: max_samples_.value() > X.rows()");
    }
    if ((max_features_.mode == MaxFeaturesMode::Count) && max_features_.count > X.cols()){
        throw DimensionMismatch("RandomForestClassifier::fit: max_features_.count > X.cols");
    }
    for (double label : y) {
    if (!std::isfinite(label)) {
        throw std::invalid_argument(
            "RandomForestClassifier::fit: non-finite class label"
        );
    }
    }

    std::vector<double> new_classes(y.begin(), y.end());

    std::sort(new_classes.begin(), new_classes.end());

    new_classes.erase(
        std::unique(new_classes.begin(), new_classes.end()),
        new_classes.end()
    );    

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

    std::vector<std::unique_ptr<CartClassifier>> new_trees(n_estimators_);

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

                auto tree = std::make_unique<CartClassifier>(max_depth_, min_samples_split_, min_samples_leaf_, max_features_, seeds[i]);
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


    trees_ = std::move(new_trees);
    classes_ = std::move(new_classes);
    num_features_ = X.cols();
    fitted_ = true;
}
//
std::vector<double> RandomForestClassifier::predict(const Matrix& X) const {
    Matrix probabilities = predict_proba(X);

    std::vector<double> result(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i) {
        std::size_t best_class = 0;

        for (std::size_t j = 1; j < classes_.size(); ++j) {
            if (probabilities(i, j) > probabilities(i, best_class)) {
                best_class = j;
            }
        }

        result[i] = classes_[best_class];
    }

    return result;
}

Matrix RandomForestClassifier::predict_proba(const Matrix& X) const {
    if (!fitted_){
    throw NotFitted("RandomForestClassifier::predict_proba: model is not fitted");
    }
    if (X.cols() != num_features_) throw FeatureCountMismatch("RandomForestClassifier::predict_proba: feature count mismatch");

    Matrix result;
    result.set_size(X.rows(), classes_.size());
    result.data().resize(X.rows() * classes_.size());
    std::fill(result.data().begin(), result.data().end(), 0.0);

    std::vector<Matrix> tree_probabilities(n_estimators_);
    std::size_t workers_count = std::min(n_jobs_, n_estimators_);
    std::vector<std::exception_ptr> errors(n_estimators_);
{
    std::vector<std::jthread> workers;
    workers.reserve(workers_count);

    for (std::size_t worker_id = 0; worker_id < workers_count;worker_id++){
        workers.emplace_back([&, worker_id]() {
            for (std::size_t i = worker_id; i < n_estimators_; i += workers_count){
                try{
                    tree_probabilities[i] = trees_[i]->predict_proba(X);
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
    
    for (std::size_t i = 0; i < n_estimators_;i++){
        auto tree_classes = trees_[i]->classes();
        for (std::size_t j = 0; j < tree_classes.size(); ++j) {

        auto it = std::lower_bound(classes_.begin(), classes_.end(), tree_classes[j]);
        std::size_t global_index = static_cast<std::size_t>(it - classes_.begin());
        for (std::size_t row = 0; row < X.rows(); ++row) {
            result(row, global_index) += tree_probabilities[i](row, j);
        }
    }

    
}
    for (double& value : result.data()) {
    value /= static_cast<double>(n_estimators_);
    }
    return result;
}

std::span<const double> RandomForestClassifier::classes() const {
    if (!fitted_) {
        throw NotFitted("RandomForestClassifier::classes: model is not fitted");
    }
    return classes_;
}
