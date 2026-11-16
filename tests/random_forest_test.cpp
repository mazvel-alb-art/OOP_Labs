#include "CartClassifier.hpp"
#include "CartRegressor.hpp"
#include "Exceptions.hpp"
#include "RandomForestClassifier.hpp"
#include "RandomForestRegressor.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {
int fail(const char* message) {
    std::cerr << "random_forest_test: FAIL: " << message << '\n';
    return 1;
}
}

int main() {
    Matrix X;
    X.set_size(8, 2);
    X.data() = {0, 0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 1, 6, 0, 7, 1};
    const std::vector<double> labels{0, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<double> targets{0, 1, 2, 3, 4, 5, 6, 7};

    RandomForestClassifier not_fitted(3);
    try {
        (void)not_fitted.predict(X);
        return fail("predict before fit should throw");
    } catch (const NotFitted&) {
    }

    RandomForestClassifier one_tree(1, 0, 2, 1,
        {MaxFeaturesMode::All, 0}, false, std::nullopt, 11, 1);
    CartClassifier cart(0, 2, 1, {MaxFeaturesMode::All, 0}, 11);
    one_tree.fit(X, labels);
    cart.fit(X, labels);
    const auto forest_probs = one_tree.predict_proba(X);
    const auto cart_probs = cart.predict_proba(X);
    if (forest_probs.rows() != X.rows() || forest_probs.cols() != cart.classes().size()) {
        return fail("probability matrix dimensions");
    }
    for (std::size_t i = 0; i < X.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < forest_probs.cols(); ++j) {
            sum += forest_probs(i, j);
            if (std::fabs(forest_probs(i, j) - cart_probs(i, j)) > 1e-12) {
                return fail("single-tree forest should match CART without bootstrap");
            }
        }
        if (std::fabs(sum - 1.0) > 1e-12) return fail("probabilities should sum to one");
    }
    if (one_tree.classes().size() != 2) return fail("classes should be exposed");

    // Один содержательный признак среди константных: дерево должно продолжить
    // просмотр случайного порядка, если первые кандидаты оказались константны.
    Matrix constants_first;
    constants_first.set_size(8, 3);
    constants_first.data() = {
        5, 9, 0, 5, 9, 1, 5, 9, 2, 5, 9, 3,
        5, 9, 4, 5, 9, 5, 5, 9, 6, 5, 9, 7
    };
    CartClassifier random_features(0, 2, 1,
        {MaxFeaturesMode::Count, 1}, 17);
    random_features.fit(constants_first, labels);
    if (random_features.predict(constants_first) != labels) {
        return fail("feature search should pass constant candidates");
    }

    Matrix zero_gain_X;
    zero_gain_X.set_size(4, 1);
    zero_gain_X.data() = {0, 1, 2, 3};
    const std::vector<double> alternating_labels{0, 1, 0, 1};
    CartClassifier zero_gain_tree(1, 2, 1, {MaxFeaturesMode::All, 0}, 3);
    zero_gain_tree.fit(zero_gain_X, alternating_labels);
    if (zero_gain_tree.node_count() != 3) {
        return fail("zero impurity improvement must not prohibit a split");
    }

    RandomForestClassifier sequential(9, 0, 2, 1,
        {MaxFeaturesMode::Sqrt, 0}, true, std::nullopt, 123, 1);
    RandomForestClassifier parallel(9, 0, 2, 1,
        {MaxFeaturesMode::Sqrt, 0}, true, std::nullopt, 123, 3);
    sequential.fit(X, labels);
    parallel.fit(X, labels);
    const auto sequential_probs = sequential.predict_proba(X);
    const auto parallel_probs = parallel.predict_proba(X);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        for (std::size_t j = 0; j < sequential_probs.cols(); ++j) {
            if (std::fabs(sequential_probs(i, j) - parallel_probs(i, j)) > 1e-12) {
                return fail("results should not depend on n_jobs");
            }
        }
    }
    sequential.fit(X, labels);
    const auto refit_probs = sequential.predict_proba(X);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        for (std::size_t j = 0; j < refit_probs.cols(); ++j) {
            if (std::fabs(sequential_probs(i, j) - refit_probs(i, j)) > 1e-12) {
                return fail("repeated fit should restart from the same seed");
            }
        }
    }

    RandomForestRegressor regressor(1, 0, 2, 1,
        {MaxFeaturesMode::All, 0}, false, std::nullopt, 11, 1);
    CartRegressor cart_regressor(0, 2, 1, {MaxFeaturesMode::All, 0}, 11);
    regressor.fit(X, targets);
    cart_regressor.fit(X, targets);
    const auto forest_predictions = regressor.predict(X);
    const auto cart_predictions = cart_regressor.predict(X);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (std::fabs(forest_predictions[i] - cart_predictions[i]) > 1e-12) {
            return fail("single-tree regressor should match CART without bootstrap");
        }
    }

    Matrix empty_queries;
    empty_queries.set_size(0, X.cols());
    const auto empty_probs = sequential.predict_proba(empty_queries);
    if (empty_probs.rows() != 0 || empty_probs.cols() != 2) {
        return fail("empty query should preserve probability matrix shape");
    }

    Matrix column_major;
    column_major.set_size(X.rows(), X.cols());
    column_major.set_layout(MatrixLayout::ColumnMajor);
    column_major.data().resize(X.rows() * X.cols());
    for (std::size_t i = 0; i < X.rows(); ++i) {
        for (std::size_t j = 0; j < X.cols(); ++j) column_major(i, j) = X(i, j);
    }
    RandomForestClassifier layout_model(5, 0, 2, 1,
        {MaxFeaturesMode::All, 0}, false, std::nullopt, 19, 1);
    layout_model.fit(column_major, labels);
    const auto layout_prediction = layout_model.predict(column_major);
    if (layout_prediction.size() != labels.size()) return fail("column-major layout prediction size");
    RandomForestClassifier row_layout_model(5, 0, 2, 1,
        {MaxFeaturesMode::All, 0}, false, std::nullopt, 19, 1);
    row_layout_model.fit(X, labels);
    if (row_layout_model.predict(X) != layout_prediction) {
        return fail("logical predictions should not depend on matrix layout");
    }

    const std::vector<double> one_class_labels(8, 4.0);
    RandomForestClassifier one_class(3, 0, 2, 1,
        {MaxFeaturesMode::Sqrt, 0}, true, std::nullopt, 8, 2);
    one_class.fit(X, one_class_labels);
    const auto one_class_probs = one_class.predict_proba(X);
    if (one_class.classes().size() != 1 || one_class_probs.cols() != 1) {
        return fail("single-class forest dimensions");
    }
    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (std::fabs(one_class_probs(i, 0) - 1.0) > 1e-12) {
            return fail("single-class probability should be one");
        }
    }

    RandomForestClassifier rare_bootstrap(1, 0, 2, 1,
        {MaxFeaturesMode::All, 0}, true, 1, 2, 1);
    rare_bootstrap.fit(X, labels);
    if (rare_bootstrap.classes().size() != 2) {
        return fail("forest class list should include every training class");
    }
    const auto rare_probs = rare_bootstrap.predict_proba(X);
    bool observed_zero_probability = false;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (rare_probs(i, 0) == 0.0 || rare_probs(i, 1) == 0.0) {
            observed_zero_probability = true;
        }
    }
    if (!observed_zero_probability) {
        return fail("class absent from bootstrap should receive zero probability");
    }

    try {
        RandomForestClassifier invalid(0);
        (void)invalid;
        return fail("zero estimators should throw");
    } catch (const std::invalid_argument&) {
    }

    std::cout << "random_forest_test: OK\n";
    return 0;
}
