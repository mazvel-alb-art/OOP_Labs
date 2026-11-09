#include "CartRegressor.hpp"
#include "Exceptions.hpp"
#include "Matrix.hpp"
#include "SplitFinder.hpp"
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>

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
CartRegressor::CartRegressor(std::size_t max_depth, std::size_t min_samples_split, std::size_t min_samples_leaf)
    : max_depth_(max_depth), min_samples_leaf_(min_samples_leaf), min_samples_split_(min_samples_split){
        if (min_samples_split_ < 2){
            throw std::invalid_argument("CartRegressor: min_samples_split must be >= 2");
        }
        if (min_samples_leaf_ < 1){
            throw std::invalid_argument("CartRegressor: min_samples_leaf must be >= 1");
        }
    }
    void CartRegressor::fit(const Matrix& X, std::span<const double> y){
    if (X.rows() == 0) {
        throw EmptyTrainingSet("CartRegressor::fit: empty training set");
    }
    if (X.rows() != y.size()) {
        throw DimensionMismatch("CartRegressor::fit: X.rows() != y.size()");
    }
    if (X.cols() == 0) {
        throw DimensionMismatch("CartRegressor::fit: X.cols() == 0");
    }
    std::vector<std::size_t> object_ids(X.rows());
    std::iota(object_ids.begin(), object_ids.end(), 0);
    root = build_node(X, y, object_ids, 0);
    num_features_ = X.cols();
    fitted = true;
    }

//
std::unique_ptr<CartNode> CartRegressor::make_leaf(std::span<const double> y, const std::vector<std::size_t>& object_ids) const{
    auto leaf = std::make_unique<CartNode>();
    leaf->is_leaf = true;
    double sum = 0.0;
    for (std::size_t idx : object_ids){
        sum += y[idx];
    }
    leaf->value = sum/static_cast<double>(object_ids.size());
    return leaf;
}
//
std::unique_ptr<CartNode> CartRegressor::build_node(const Matrix& X, std::span<const double> y, const std::vector<std::size_t>& object_ids, std::size_t depth){
    if (depth >= max_depth_){
        return make_leaf(y, object_ids);
    }
    if (object_ids.size() < min_samples_split_){
        return make_leaf(y, object_ids);
    }
    bool all_same = true;
    for (std::size_t idx : object_ids){
        if (y[idx] != y[object_ids[0]]){
            all_same = false;
            break;
        }
    }
    if (all_same == true){
        return make_leaf(y, object_ids);
    }
    std::vector<std::size_t> feature_ids(X.cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    SplitFinder finder;
    auto split = finder.find_best(X, y, object_ids, feature_ids, SplitCriterion::Variance, min_samples_leaf_);

    if (split == std::nullopt){
        return make_leaf(y, object_ids);
    }

    std::vector<std::size_t> left_ids, right_ids;
    for (std::size_t idx : object_ids){
        if (X(idx, split->feature) <= split->threshold){
            left_ids.push_back(idx);
        }
        else{
            right_ids.push_back(idx);
        }
    }
    auto node = std::make_unique<CartNode>();
    node->is_leaf = false;
    node->feature = split->feature;
    node->threshold = split->threshold;
    node->left = build_node(X, y, left_ids, depth + 1);
    node->right = build_node(X, y, right_ids, depth + 1);
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