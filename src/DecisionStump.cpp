#include "DecisionStump.hpp"
#include "SplitFinder.hpp"
#include "Exceptions.hpp"

#include <numeric>
#include <algorithm>
#include <vector>
#include <limits>
#include <map>
#include <set>

DecisionStump::DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf) {
    if (min_samples_leaf < 1) {
        throw std::invalid_argument("Неверное минимальное количество объектов для каждой стороны");
    }
    criterion_ = criterion;
    min_samples_leaf_ = min_samples_leaf;
}

void DecisionStump::fit(const Matrix& features, std::span<const double> target) {
    // проверки
    if (features.rows() == 0) {
        throw EmptyTrainingSet("Пустая матрица");
    }
    if (features.rows() != target.size()) {
        throw DimensionMismatch("Несоответствие объектов и откликов");
    }

    train_cols_ = features.cols();

    // Если объектов слишком мало для разбиения — константа
    if (features.rows() < 2 * min_samples_leaf_) {
        has_split_ = false;
        if (criterion_ == SplitCriterion::Variance) {
            double mean = 0.0;
            for (double v : target) mean += v;
            left_value_ = right_value_ = mean / static_cast<double>(target.size());
        } else {
            // самый частый класс
            std::map<double, std::size_t> counts;
            for (double v : target) counts[v]++;
            double best_label = target[0];
            std::size_t best_count = 0;
            for (auto& [label, cnt] : counts) {
                if (cnt > best_count) {
                    best_count = cnt;
                    best_label = label;
                }
            }
            left_value_ = right_value_ = best_label;
        }
        fitted_ = true;
        return;
    }

    // Все индексы объектов и признаков
    std::vector<std::size_t> object_ids(features.rows());
    std::iota(object_ids.begin(), object_ids.end(), 0);

    std::vector<std::size_t> feature_ids(features.cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    // Поиск лучшего разбиения через общий механизм
    SplitFinder finder;
    auto split = finder.find_best(features, target, object_ids, feature_ids,
                                   criterion_, min_samples_leaf_);

    if (!split.has_value()) {
        has_split_ = false;
        if (criterion_ == SplitCriterion::Variance) {
            double mean = 0.0;
            for (double v : target) mean += v;
            left_value_ = right_value_ = mean / static_cast<double>(target.size());
        } else {
            std::map<double, std::size_t> counts;
            for (double v : target) counts[v]++;
            double best_label = target[0];
            std::size_t best_count = 0;
            for (auto& [label, cnt] : counts) {
                if (cnt > best_count) {
                    best_count = cnt;
                    best_label = label;
                }
            }
            left_value_ = right_value_ = best_label;
        }
        fitted_ = true;
        return;
    }

    has_split_ = true;
    split_feature_ = split->feature;
    split_threshold_ = split->threshold;

    // Значения в листьях — по объектам, попавшим в каждую часть
    if (criterion_ == SplitCriterion::Variance) {
        double sum_L = 0.0, sum_R = 0.0;
        std::size_t cnt_L = 0, cnt_R = 0;
        for (std::size_t i = 0; i < features.rows(); ++i) {
            if (features(i, split_feature_) <= split_threshold_) {
                sum_L += target[i];
                ++cnt_L;
            } else {
                sum_R += target[i];
                ++cnt_R;
            }
        }
        left_value_ = sum_L / static_cast<double>(cnt_L);
        right_value_ = sum_R / static_cast<double>(cnt_R);
    } else {
        std::map<double, std::size_t> count_L, count_R;
        for (std::size_t i = 0; i < features.rows(); ++i) {
            if (features(i, split_feature_) <= split_threshold_) {
                count_L[target[i]]++;
            } else {
                count_R[target[i]]++;
            }
        }
        // самый частый в каждой части (при равенстве — меньший индекс)
        double best_L = 0.0; std::size_t best_L_count = 0;
        for (auto& [label, cnt] : count_L) {
            if (cnt > best_L_count) { best_L_count = cnt; best_L = label; }
        }
        double best_R = 0.0; std::size_t best_R_count = 0;
        for (auto& [label, cnt] : count_R) {
            if (cnt > best_R_count) { best_R_count = cnt; best_R = label; }
        }
        left_value_ = best_L;
        right_value_ = best_R;
    }

    fitted_ = true;
}

std::vector<double> DecisionStump::predict(const Matrix& features) const {
    if (!fitted_) {
        throw NotFitted("fit пропущен!");
    }
    if (features.cols() != train_cols_) {
        throw FeatureCountMismatch("Несоответствие количества признаков");
    }
    std::size_t n = features.rows();
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!has_split_) {
            result[i] = left_value_;
        } else if (features(i, split_feature_) <= split_threshold_) {
            result[i] = left_value_;
        } else {
            result[i] = right_value_;
        }
    }
    return result;
}