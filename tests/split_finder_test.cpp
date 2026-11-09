#include "SplitFinder.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    // Линейная зависимость: y = 2x. Лучший порог — где-то посередине.
    Matrix X;
    X.set_size(6, 1);
    X.data() = {1, 2, 3, 4, 5, 6};

    std::vector<double> y = {2, 4, 6, 8, 10, 12};

    std::vector<std::size_t> obj = {0, 1, 2, 3, 4, 5};
    std::vector<std::size_t> feat = {0};

    SplitFinder f;
    auto split = f.find_best(X, y, obj, feat, SplitCriterion::Variance, 1);

    if (!split.has_value()) {
        std::cout << "split_finder_test: no split found\n";
        return 1;
    }
    if (split->feature != 0) {
        std::cout << "split_finder_test: wrong feature\n";
        return 1;
    }
    if (split->threshold <= 1.0 || split->threshold >= 6.0) {
        std::cout << "split_finder_test: threshold out of range: "
                  << split->threshold << "\n";
        return 1;
    }

    // Gini: классы 0,0,0,1,1,1
    std::vector<double> yc = {0, 0, 0, 1, 1, 1};
    auto split2 = f.find_best(X, yc, obj, feat, SplitCriterion::Gini, 1);
    if (!split2.has_value()) {
        std::cout << "split_finder_test: no gini split\n";
        return 1;
    }
    if (split2->threshold < 2.0 || split2->threshold > 4.0) {
        std::cout << "split_finder_test: gini threshold out of range: "
                  << split2->threshold << "\n";
        return 1;
    }

    std::cout << "split_finder_test: OK\n";
    return 0;
}
