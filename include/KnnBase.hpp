#ifndef KNNBASE_HPP
#define KNNBASE_HPP

#include "IDistance.hpp"
#include "Matrix.hpp"
#include "IModel.hpp"

#include <memory>
#include <span>
#include <vector>
#include <utility>
#include <cstddef>

/**
 * @brief Способ взвешивания соседей при предсказании.
 */
enum class Weighting { Uniform, Distance };

/**
 * @brief Общая часть для моделей kNN (регрессор и классификатор).
 *
 * Содержит: метрику, k, способ взвешивания, ссылку на обучающую выборку,
 * метод fit и метод поиска соседей. Метод predict не реализован —
 * он остаётся наследникам (KnnRegressor, KnnClassifier).
 *
 * Модель не копирует обучающие данные: хранит указатель на матрицу
 * и span на отклики. Данные должны жить дольше модели.
 */
class KnnBase : public IModel
{
    protected:
    std::unique_ptr<IDistance> metric_;
    std::size_t k_;
    const Matrix* train_features_ = nullptr;
    std::span<const double> train_target_;
    Weighting weighting_;

    public:
    /**
     * @brief Конструктор.
     * @param metric Метрика расстояния (модель забирает владение).
     * @param k Число соседей (k >= 1).
     * @param weighting Способ взвешивания соседей.
     * @throw InvalidK Если k < 1.
     */
    KnnBase(std::unique_ptr<IDistance> metric, std::size_t k, Weighting weighting);
    virtual ~KnnBase() = default;

    /**
     * @brief Запоминает обучающую выборку.
     * @param features Матрица признаков train.
     * @param target Вектор откликов train.
     * @throw EmptyTrainingSet Если features.rows() == 0.
     * @throw DimensionMismatch Если features.rows() != target.size().
     */
    void fit(const Matrix& features, std::span<const double> target) override;

    /**
     * @brief Ищет k ближайших соседей запроса.
     * @param query Вектор признаков запроса.
     * @param k Число соседей.
     * @return Вектор пар (расстояние, индекс в train), отсортированный по расстоянию.
     * @throw NotFitted Если fit не вызывался.
     * @throw InvalidK Если k == 0 или k > размера train.
     * @throw DimensionMismatch Если query.size() != train.cols().
     */
    std::vector<std::pair<double, std::size_t>> find_neighbors(std::span<const double> query, std::size_t k) const;
};

#endif