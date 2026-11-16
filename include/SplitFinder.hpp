#ifndef SPLITFINDER_HPP
#define SPLITFINDER_HPP

#include "Matrix.hpp"
#include "DecisionStump.hpp"  // там enum SplitCriterion
#include <span>
#include <optional>
#include <cstddef>

/**
 * @brief Описание найденного разбиения.
 */
struct Split {
    std::size_t feature;   ///< Номер признака (столбца).
    double threshold;      ///< Порог.
    double score;          ///< Значение критерия.
};

/** @brief Результат поиска разбиения по одному признаку. */
struct FeatureSplitResult {
    bool is_constant = true; ///< Значения признака у объектов узла одинаковы.
    std::optional<Split> best; ///< Лучшее допустимое разбиение или std::nullopt.
};

/**
 * @brief Поиск лучшего разбиения.
 *
 * Работает на подмножестве объектов (object_ids) и подмножестве признаков
 * (feature_ids). Возвращает std::nullopt, если допустимого разбиения нет.
 */
class SplitFinder {
public:
    /**
     * @brief Найти лучшее разбиение.
     * @param X Матрица признаков.
     * @param y Вектор откликов, индексируемый исходными номерами строк.
     * @param object_ids Индексы объектов, которые участвуют в разбиении.
     * @param feature_ids Индексы разрешённых признаков.
     * @param criterion Критерий (Variance или Gini).
     * @param min_samples_leaf Минимум объектов в каждой части.
     * @param counts Кратности исходных строк; пустой span означает единичные веса.
     * @return `Split` или `std::nullopt`, если разбиения нет.
     */
    std::optional<Split> find_best(
        const Matrix& X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        SplitCriterion criterion,
        std::size_t min_samples_leaf,
        std::span<const std::size_t> counts = {}
    ) const;
    
    /**
     * @brief Найти лучшее разбиение по одному признаку.
     * @param X Матрица признаков.
     * @param y Вектор откликов, индексируемый исходными номерами строк.
     * @param object_ids Номера строк, входящих в текущий узел.
     * @param feature_id Номер проверяемого столбца.
     * @param criterion Критерий разбиения.
     * @param min_samples_leaf Минимальное число разных объектов в каждой части.
     * @param counts Кратности исходных строк; пустой span означает единичные веса.
     * @return Признак константности и лучшее допустимое разбиение.
     */
    FeatureSplitResult find_for_feature(
        const Matrix& X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::size_t feature_id,
        SplitCriterion criterion,
        std::size_t min_samples_leaf,
        std::span<const std::size_t> counts = {}
    ) const;
};

#endif
