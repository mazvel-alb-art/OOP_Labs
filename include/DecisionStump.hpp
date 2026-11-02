#ifndef DECISIONSTUMP_HPP
#define DECISIONSTUMP_HPP

#include "IModel.hpp"
#include <span>
#include <vector>
#include <cstddef>

/**
 * @brief Критерий разбиения для решающего пня.
 *
 * - Variance — суммарная ошибка предсказания константой (SSE, регрессия).
 * - Gini — индекс Джини (классификация).
 */
enum class SplitCriterion { Variance, Gini };

/**
 * @brief Решающий пень — дерево решений глубины 1.
 *
 * Модель выбирает один признак j и один порог t. Объекты с
 * X[i][j] <= t попадают в левую часть, остальные — в правую.
 * В каждой части предсказывается константа:
 * - для регрессии — среднее отклика в части;
 * - для классификации — самый частый класс.
 *
 * Обучение — перебор всех (j, t) с инкрементальным обновлением
 * агрегатов за O(1). Сложность O(d * n log n), память O(n).
 */
class DecisionStump : public IModel{
private:
    SplitCriterion criterion_;
    std::size_t min_samples_leaf_;
    std::size_t train_cols_ = 0;

    bool fitted_ = false;
    bool has_split_ = false;

    std::size_t split_feature_ = 0;
    double split_threshold_ = 0.0;
    double left_value_ = 0.0;
    double right_value_ = 0.0;

    /**
     * @brief Обучение для регрессии (критерий Variance).
     * @param features Матрица признаков.
     * @param target Вектор числовых откликов.
     */
    void fit_variance(const Matrix& features, std::span<const double> target);

    /**
     * @brief Обучение для классификации (критерий Gini).
     * @param features Матрица признаков.
     * @param target Вектор числовых кодов классов.
     */
    void fit_gini(const Matrix& features, std::span<const double> target);

public:
    /**
     * @brief Конструктор.
     * @param criterion Критерий разбиения (Variance или Gini).
     * @param min_samples_leaf Минимум объектов в каждой части (>= 1).
     * @throw std::invalid_argument Если min_samples_leaf < 1.
     */
    DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf);
    virtual ~DecisionStump() = default;

    /**
     * @brief Обучает пень на обучающей выборке.
     * @param features Матрица признаков (объекты — строки).
     * @param target Вектор откликов, target[i] соответствует features[i].
     * @throw EmptyTrainingSet Если features.rows() == 0.
     * @throw DimensionMismatch Если features.rows() != target.size().
     */
    void fit(const Matrix& features, std::span<const double> target) override;

    /**
     * @brief Предсказывает отклики для объектов.
     * @param features Матрица признаков.
     * @return Вектор предсказаний длины features.rows().
     * @throw NotFitted Если fit не вызывался.
     * @throw FeatureCountMismatch Если features.cols() != train_cols_.
     */
    std::vector<double> predict(const Matrix& features) const override;

    /**
     * @brief Номер выбранного признака (j).
     */
    std::size_t split_feature() const{return split_feature_;}

    /**
     * @brief Выбранный порог (t).
     */
    double split_threshold() const{return split_threshold_;}

    /**
     * @brief Константа левой части.
     */
    double left_value() const{return left_value_;}

    /**
     * @brief Константа правой части.
     */
    double right_value() const{return right_value_;}

    /**
     * @brief Был ли найден валидный порог при fit.
     */
    bool has_split() const{return has_split_;}
};

#endif