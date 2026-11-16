#include "CartClassifier.hpp"
#include "Dataset.hpp"
#include "DecisionStump.hpp"
#include "RandomForestClassifier.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {

double accuracy(const std::vector<double>& prediction,
                const std::vector<double>& truth) {
    if (prediction.size() != truth.size() || truth.empty()) {
        return 0.0;
    }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (prediction[i] == truth[i]) {
            ++correct;
        }
    }
    return 100.0 * static_cast<double>(correct) /
           static_cast<double>(truth.size());
}

template <typename Model>
void report_model(const char* title, Model& model, const Split& split) {
    const auto fit_start = std::chrono::steady_clock::now();
    model.fit(split.train_X, std::span<const double>(split.train_y));
    const auto fit_end = std::chrono::steady_clock::now();

    const auto predict_start = std::chrono::steady_clock::now();
    const auto prediction = model.predict(split.test_X);
    const auto predict_end = std::chrono::steady_clock::now();

    const auto fit_ms = std::chrono::duration<double, std::milli>(fit_end - fit_start).count();
    const auto predict_ms = std::chrono::duration<double, std::milli>(predict_end - predict_start).count();
    std::cout << title << "\n"
              << "  Accuracy: " << accuracy(prediction, split.test_y) << "%\n"
              << "  Fit: " << fit_ms << " ms; predict: " << predict_ms << " ms\n";
}

} // namespace

int main() {
    try {
        Dataset dataset("data/crx.csv");
        dataset.preprocess();
        const Split split = dataset.train_test_split(0.8);

        std::cout << "Lab 4 — классификация CRX\n"
                  << "Train: " << split.train_X.rows()
                  << ", test: " << split.test_X.rows()
                  << ", features: " << split.train_X.cols() << "\n"
                  << "Одинаковое детерминированное разбиение для всех моделей.\n\n";
        std::cout << std::fixed << std::setprecision(2);

        DecisionStump stump(SplitCriterion::Gini, 1);
        report_model("Decision stump", stump, split);

        CartClassifier tree(0, 2, 1, {MaxFeaturesMode::All, 0}, 42);
        report_model("CART tree (unlimited depth)", tree, split);
        std::cout << "  Nodes: " << tree.node_count()
                  << ", leaves: " << tree.leaf_count()
                  << ", depth: " << tree.actual_depth() << "\n\n";

        constexpr std::size_t estimators = 50;
        RandomForestClassifier forest(
            estimators, 0, 2, 1, {MaxFeaturesMode::Sqrt, 0},
            true, std::nullopt, 42, 1);
        report_model("Random forest", forest, split);
        std::cout << "  Trees: " << estimators
                  << ", max_features: sqrt, bootstrap: yes, n_jobs: 1\n";

        const auto probabilities = forest.predict_proba(split.test_X);
        std::cout << "  First predictions (pred / true / class probabilities):\n";
        const auto predictions = forest.predict(split.test_X);
        const std::size_t count = std::min<std::size_t>(5, predictions.size());
        for (std::size_t i = 0; i < count; ++i) {
            std::cout << "    " << predictions[i] << " / " << split.test_y[i] << " / [";
            for (std::size_t c = 0; c < probabilities.cols(); ++c) {
                if (c != 0) std::cout << ", ";
                std::cout << probabilities(i, c);
            }
            std::cout << "]\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Lab4 error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
