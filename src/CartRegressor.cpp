#include "CartRegressor.hpp"
#include "Exceptions.hpp"
#include "Matrix.hpp"
#include "SplitFinder.hpp"
#include "MaxFeatures.hpp"
#include <memory>
#include <numeric>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

namespace {
    std::size_t count_nodes(const CartNode* node) {
        if (node == nullptr) return 0;
        if (node->is_leaf) return 1;
        return 1 + count_nodes(node->left.get()) + count_nodes(node->right.get());
    }

    std::size_t count_leaves(const CartNode* node) {
        if (node == nullptr) return 0;
        if (node->is_leaf) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

    std::size_t tree_depth(const CartNode* node) {
        if (node == nullptr) return 0;
        if (node->is_leaf) return 0;
        return 1 + std::max(tree_depth(node->left.get()), tree_depth(node->right.get()));
    }
}

std::size_t CartRegressor::node_count() const {
    if (!fitted) throw NotFitted("CartRegressor::node_count: model is not fitted");
    return count_nodes(root.get());
}

std::size_t CartRegressor::leaf_count() const {
    if (!fitted) throw NotFitted("CartRegressor::leaf_count: model is not fitted");
    return count_leaves(root.get());
}

std::size_t CartRegressor::actual_depth() const {
    if (!fitted) throw NotFitted("CartRegressor::actual_depth: model is not fitted");
    return tree_depth(root.get());
}
//
CartRegressor::CartRegressor(std::size_t max_depth, std::size_t min_samples_split, std::size_t min_samples_leaf, MaxFeatures max_features, std::uint32_t random_state)
    : max_depth_(max_depth), min_samples_leaf_(min_samples_leaf), min_samples_split_(min_samples_split), max_features_(max_features), random_state_(random_state){
        if (min_samples_split_ < 2){
            throw std::invalid_argument("CartRegressor: min_samples_split must be >= 2");
        }
        if (min_samples_leaf_ < 1){
            throw std::invalid_argument("CartRegressor: min_samples_leaf must be >= 1");
        }
    }
//
void CartRegressor::fit(const Matrix& X, std::span<const double> y){
    std::vector<std::size_t> counts(y.size(), 1);
    fit_with_counts(X, y, counts);
}
//
void CartRegressor::fit_with_counts(const Matrix& X, std::span<const double> y, std::span<const std::size_t> counts){
    if (X.rows() == 0) {
        throw EmptyTrainingSet("CartRegressor::fit: empty training set");
    }
    if (X.rows() != y.size()) {
        throw DimensionMismatch("CartRegressor::fit: X.rows() != y.size()");
    }
    if (X.cols() == 0) {
        throw DimensionMismatch("CartRegressor::fit: X.cols() == 0");
    }
    if (counts.size() != y.size()){
        throw DimensionMismatch("CartRegressor::fit: counts.size() != y.size()");
    }
    if (max_features_.mode == MaxFeaturesMode::Count && (max_features_.count == 0 || max_features_.count > X.cols())){
        throw std::invalid_argument("CartRegressor: invalid max_features count");
    }
    bool empty = true;
    for (std::size_t i = 0; i < counts.size();i++){
        if (counts[i] != 0){
            empty = false;
            break;
        }
    }
    if (empty){
        throw EmptyTrainingSet("CartRegressor::fit_with_counts: all counts are zero");
    }
    std::vector<std::size_t> object_ids;
    object_ids.reserve(X.rows());
    for (std::size_t i = 0; i < counts.size();i++){
        if (counts[i] > 0){
            object_ids.push_back(i);
        }
    }
    std::mt19937 rng(random_state_);
    root = build_node(X, y, object_ids, 0, counts, rng);
    num_features_ = X.cols();
    fitted = true;

}
//
std::unique_ptr<CartNode> CartRegressor::make_leaf(std::span<const double> y, const std::vector<std::size_t>& object_ids, std::span<const std::size_t> counts) const{
    double weighted_sum = 0.0;
    std::size_t total_weight = 0;
    for (std::size_t i = 0; i < object_ids.size();i++){
        total_weight += counts[object_ids[i]];
        weighted_sum += y[object_ids[i]] * counts[object_ids[i]];
    }

    auto leaf = std::make_unique<CartNode>();
    leaf->is_leaf = true;
    leaf->value = weighted_sum/static_cast<double>(total_weight);
    return leaf;
}
//

std::unique_ptr<CartNode> CartRegressor::build_node(const Matrix& X, std::span<const double> y, const std::vector<std::size_t>& object_ids, std::size_t depth, std::span<const std::size_t> counts, std::mt19937& rng){
    if (max_depth_ != 0 && depth >= max_depth_){
        return make_leaf(y, object_ids, counts);
    }
    if (object_ids.size() < min_samples_split_){
        return make_leaf(y, object_ids, counts);
    }
    bool all_same = true;
    for (std::size_t idx : object_ids){
        if (y[idx] != y[object_ids[0]]){
            all_same = false;
            break;
        }
    }
    if (all_same == true){
        return make_leaf(y, object_ids, counts);
    }
    std::vector<std::size_t> feature_ids(X.cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0);
    std::shuffle(feature_ids.begin(), feature_ids.end(), rng);
    std::size_t g;
    if (max_features_.mode == MaxFeaturesMode::All ){
        g = X.cols();
    }
    else if (max_features_.mode == MaxFeaturesMode::Sqrt){
        g = static_cast<std::size_t>(std::sqrt(static_cast<double>(X.cols())));
    }
    else{
        g = max_features_.count;
    }


    SplitFinder finder;
    std::optional<Split> best_split;
    bool all_constant = true;

    
    for (std::size_t i = 0; i < g ; i++){
        FeatureSplitResult result = finder.find_for_feature(X, y, object_ids, feature_ids[i], SplitCriterion::Variance, min_samples_leaf_, counts);
    if (!(result.is_constant)){
        all_constant = false;
    }
    if (!best_split.has_value()){
        best_split = result.best;
        continue;
    }
    if (result.best.has_value()){
        if (best_split->score < result.best->score){
            best_split = result.best;
        }
    }
    }
    if (all_constant){
        for (std::size_t i = g; i < feature_ids.size() ; i++){
            FeatureSplitResult result = finder.find_for_feature(X, y, object_ids, feature_ids[i], SplitCriterion::Variance, min_samples_leaf_, counts);
            if (!(result.is_constant)){
                if (result.best.has_value()){
                    best_split = result.best;
                }
                break;
            }
        }
    }


    if (!best_split.has_value()){
        return make_leaf(y, object_ids, counts);
    }

    std::vector<std::size_t> left_ids, right_ids;
    for (std::size_t idx : object_ids){
        if (X(idx, best_split->feature) <= best_split->threshold){
            left_ids.push_back(idx);
        }
        else{
            right_ids.push_back(idx);
        }
    }
    auto node = std::make_unique<CartNode>();
    node->is_leaf = false;
    node->feature = best_split->feature;
    node->threshold = best_split->threshold;
    node->left = build_node(X, y, left_ids, depth + 1, counts, rng);
    node->right = build_node(X, y, right_ids, depth + 1, counts, rng);
    return node;
}

double CartRegressor::predict_one(MatrixSlice<const double> row) const{
    const CartNode* node = root.get();
    while (!node->is_leaf) {
        if (row(node->feature) <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->value;
}
std::vector<double> CartRegressor::predict(const Matrix& X) const{
    if (!fitted) {
        throw NotFitted("CartRegressor::predict: model is not fitted");
    }
    if (X.cols() != num_features_) {
        throw FeatureCountMismatch("CartRegressor::predict: feature count mismatch");
    }
    std::vector<double> result(X.rows());
    for (std::size_t i = 0; i < X.rows();i++){
        result[i] = predict_one(X(i, std::nullopt));
    }
    return result;
}