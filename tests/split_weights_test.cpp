#include "SplitFinder.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {
int fail(const char* message) {
    std::cerr << "split_weights_test: FAIL: " << message << '\n';
    return 1;
}
}

int main() {
    Matrix X;
    X.set_size(3, 1);
    X.data() = {0.0, 1.0, 2.0};
    const std::vector<double> regression_y{0.0, 10.0, 10.0};
    const std::vector<double> classification_y{0.0, 1.0, 1.0};
    const std::vector<std::size_t> ids{0, 1, 2};
    const std::vector<std::size_t> features{0};
    const std::vector<std::size_t> weights{3, 1, 1};
    SplitFinder finder;

    const auto regression = finder.find_best(
        X, regression_y, ids, features, SplitCriterion::Variance, 1, weights);
    if (!regression || regression->feature != 0 ||
        std::fabs(regression->threshold - 0.5) > 1e-12 ||
        std::fabs(regression->score - 200.0) > 1e-12) {
        return fail("weighted regression criterion and threshold");
    }

    const auto classification = finder.find_best(
        X, classification_y, ids, features, SplitCriterion::Gini, 1, weights);
    if (!classification || classification->feature != 0 ||
        std::fabs(classification->threshold - 0.5) > 1e-12 ||
        std::fabs(classification->score - 5.0) > 1e-12) {
        return fail("weighted classification criterion and threshold");
    }

    const auto too_many_per_leaf = finder.find_best(
        X, regression_y, ids, features, SplitCriterion::Variance, 3, weights);
    if (too_many_per_leaf) return fail("min_samples_leaf counts distinct rows");

    const auto unweighted = finder.find_best(
        X, regression_y, ids, features, SplitCriterion::Variance, 1);
    if (!unweighted || std::fabs(unweighted->threshold - 0.5) > 1e-12) {
        return fail("empty counts span means unit weights");
    }

    std::cout << "split_weights_test: OK\n";
    return 0;
}
