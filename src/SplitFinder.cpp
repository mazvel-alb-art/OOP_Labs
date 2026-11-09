#include "SplitFinder.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

std::optional<Split> SplitFinder::find_best(
    const Matrix& X,
    std::span<const double> y,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    SplitCriterion criterion,
    std::size_t min_samples_leaf
) const {
    std::size_t n = object_ids.size();
    if (n == 0 || feature_ids.empty()) return std::nullopt;
    if (n < 2 * min_samples_leaf) return std::nullopt;

    if (criterion == SplitCriterion::Variance) {
        double S_total = 0.0;
        for (std::size_t idx : object_ids) S_total += y[idx];

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

            std::size_t m_L = 0;
            double S_L = 0.0;
            std::size_t m_R = n;
            double S_R = S_total;

            for (std::size_t p = 0; p < n - 1; ++p) {
                m_L++;
                S_L += y[order[p]];
                m_R--;
                S_R -= y[order[p]];

                if (X(order[p], j) == X(order[p + 1], j)) continue;
                if (m_L < min_samples_leaf || m_R < min_samples_leaf) continue;

                double score = (S_L * S_L) / static_cast<double>(m_L)
                             + (S_R * S_R) / static_cast<double>(m_R);

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
        std::vector<double> class_value;
        for (std::size_t idx : object_ids) {
            if (class_map.find(y[idx]) == class_map.end()) {
                class_map[y[idx]] = class_map.size();
                class_value.push_back(y[idx]);
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
            counts_total[class_of_global[idx]]++;
        }
        std::size_t A_total = 0;
        for (std::size_t i = 0; i < k; ++i) {
            A_total += counts_total[i] * counts_total[i];
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
            std::size_t A_L = 0;
            std::size_t A_R = A_total;
            std::size_t m_L = 0;
            std::size_t m_R = n;

            for (std::size_t p = 0; p < n - 1; ++p) {
                std::size_t c = class_of_global[order[p]];

                A_L += 2 * counts_L[c] + 1;
                A_R -= 2 * counts_R[c] - 1;
                m_L++;
                m_R--;
                counts_L[c]++;
                counts_R[c]--;

                if (X(order[p], j) == X(order[p + 1], j)) continue;
                if (m_L < min_samples_leaf || m_R < min_samples_leaf) continue;

                double score = static_cast<double>(A_L) / static_cast<double>(m_L)
                             + static_cast<double>(A_R) / static_cast<double>(m_R);

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
}