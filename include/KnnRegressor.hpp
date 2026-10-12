#ifndef KNNREGRESSOR_HPP
#define KNNREGRESSOR_HPP

#include "KnnBase.hpp"

/**
 * @brief kNN для задачи регрессии.
 *
 * predict возвращает среднее (или взвешенное среднее) откликов
 * k ближайших соседей.
 */
class KnnRegressor : public KnnBase
{
    public:
    /**
     * @brief Конструктор.
     * @param metric Метрика расстояния.
     * @param k Число соседей.
     * @param weighting Способ взвешивания.
     */
    KnnRegressor(std::unique_ptr<IDistance> metric, std::size_t k, Weighting weighting);

    /**
     * @brief Предсказывает числовой отклик для каждого объекта.
     * @param features Матрица признаков.
     * @return Вектор предсказаний.
     * @throw NotFitted Если fit не вызывался.
     * @throw FeatureCountMismatch Если число признаков не совпадает с train.
     */
    std::vector<double> predict(const Matrix& features) const override;
};

#endif