#include "SplitFinder.hpp"
#include "MatrixSlice.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>

std::optional<Split> SplitFinder::find_best(
    const Matrix& X,
    std::span<const double> y,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    SplitCriterion criterion,
    std::size_t min_samples_leaf, 
    std::span<const std::size_t> counts
) const {
    std::size_t n = object_ids.size();
    if (n == 0 || feature_ids.empty()) return std::nullopt;
    if (n < 2 * min_samples_leaf) return std::nullopt;

    if (criterion == SplitCriterion::Variance) {

        double best_score = -std::numeric_limits<double>::infinity();
        std::size_t best_j = 0;
        double best_t = 0.0;
        bool found = false;

        std::vector<std::size_t> order(n);

        for (std::size_t j : feature_ids) {
            for (std::size_t k = 0; k < n; ++k) order[k] = object_ids[k];
            std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
                return X(a, j) < X(b, j);
            });

            double left_sum = 0.0;
            double right_sum = 0.0;

            double left_weight = 0.0;
            double right_weight = 0.0;
            for (std::size_t idx : object_ids) {
            double weight = counts.empty() ? 1.0 : static_cast<double>(counts[idx]);
            right_weight += weight;
            right_sum += y[idx] * weight;
            }

            for (std::size_t p = 0; p < n - 1; ++p) {
                std::size_t idx = order[p];
                double weight = counts.empty() ? 1.0 : static_cast<double>(counts[idx]);
                left_weight += weight;
                left_sum += y[idx] * weight;
                right_weight -= weight;
                right_sum -= y[idx] * weight;

                if (X(order[p], j) == X(order[p + 1], j)) continue;
                if (p + 1 < min_samples_leaf || n - p - 1 < min_samples_leaf) continue;

                double score = (left_sum * left_sum) / static_cast<double>(left_weight)
                             + (right_sum * right_sum) / static_cast<double>(right_weight);

                double a = X(order[p], j);
                double b = X(order[p + 1], j);
                double t = std::midpoint(a, b);
                if (t == b) t = a;

                if (score > best_score) {
                    best_score = score; best_j = j; best_t = t; found = true;
                }
            }
        }
        if (found) return Split{best_j, best_t, best_score};
        return std::nullopt;
    }
    else {
        // Классификация
        std::map<double, std::size_t> class_map;
        for (std::size_t idx : object_ids) {
            if (class_map.find(y[idx]) == class_map.end()) {
                class_map[y[idx]] = class_map.size();
            }
        }
        std::size_t k = class_map.size();

        // class_of_global[idx] = номер класса для объекта с ГЛОБАЛЬНЫМ индексом idx
        std::unordered_map<std::size_t, std::size_t> class_of_global;
        for (std::size_t idx : object_ids) {
            class_of_global[idx] = class_map[y[idx]];
        }

        std::vector<std::size_t> counts_total(k, 0);
        for (std::size_t idx : object_ids) {
            std::size_t w = counts.empty() ? 1 : counts[idx];
            counts_total[class_of_global[idx]] += w;
        }
        double A_total = 0;
        for (std::size_t i = 0; i < k; ++i) {
            double c = static_cast<double>(counts_total[i]);
            A_total += c * c;
        }

        double best_score = -std::numeric_limits<double>::infinity();
        std::size_t best_j = 0;
        double best_t = 0.0;
        bool found = false;

        std::vector<std::size_t> order(n);

        for (std::size_t j : feature_ids) {
            for (std::size_t q = 0; q < n; ++q) order[q] = object_ids[q];
            std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
                return X(a, j) < X(b, j);
            });

            std::vector<std::size_t> counts_L(k, 0);
            std::vector<std::size_t> counts_R = counts_total;
            double A_L = 0;
            double A_R = A_total;
            std::size_t W_L = 0;
            std::size_t W_R = 0;

            for (std::size_t idx : object_ids) {
            W_R += counts.empty() ? 1 : counts[idx];
}

            for (std::size_t p = 0; p < n - 1; ++p) {
                std::size_t idx = order[p];
                std::size_t c = class_of_global[idx];
                std::size_t w = counts.empty() ? 1 : counts[idx];

                double wd = static_cast<double>(w);
                double a = static_cast<double>(counts_L[c]);
                double b = static_cast<double>(counts_R[c]);

                A_L += 2.0 * a * wd + wd * wd;
                A_R += -2.0 * b * wd + wd * wd;

                counts_L[c] += w;
                counts_R[c] -= w;
                W_L += w;
                W_R -= w;

                if (X(order[p], j) == X(order[p + 1], j)) continue;
                if (p + 1 < min_samples_leaf || n - p - 1 < min_samples_leaf) continue;

                double score = A_L / static_cast<double>(W_L)
                             + A_R / static_cast<double>(W_R);
                double a_1 = X(order[p], j);
                double b_1 = X(order[p + 1], j);
                double t = std::midpoint(a_1, b_1);
                if (t == b_1) t = a_1;

                if (score > best_score) {
                    best_score = score; best_j = j; best_t = t; found = true;
                }
            }
        }
        if (found) return Split{best_j, best_t, best_score};
        return std::nullopt;
    }
}
//
FeatureSplitResult SplitFinder::find_for_feature(
    const Matrix& X,
    std::span<const double> y,
    std::span<const std::size_t> object_ids,
    std::size_t feature_id,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts
) const{
    if ( object_ids.empty()){
        return FeatureSplitResult{false, std::nullopt};
    }
    bool constant = true;
    for (std::size_t i = 0;i < object_ids.size();i++){
        if (X(object_ids[i], feature_id) != X(object_ids[0], feature_id)){
            constant = false;
            break;
        }
    }
    if (constant){
        return FeatureSplitResult{true, std::nullopt};
    }

    return FeatureSplitResult{false, (find_best(X, y,object_ids, std::vector<std::size_t>(1, feature_id), criterion, min_samples_leaf, counts ))};
}