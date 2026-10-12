#ifndef KNNCLASSIFIER_HPP
#define KNNCLASSIFIER_HPP

#include "KnnBase.hpp"

/**
 * @brief kNN для задачи классификации.
 *
 * predict возвращает наиболее частый (или взвешенный по расстоянию)
 * класс среди k ближайших соседей. Классы кодируются числами.
 * При равных голосах побеждает класс с меньшим кодом.
 */
class KnnClassifier : public KnnBase
{
    public:
    /**
     * @brief Конструктор.
     * @param metric Метрика расстояния.
     * @param k Число соседей.
     * @param weighting Способ взвешивания.
     */
    KnnClassifier(std::unique_ptr<IDistance> metric, std::size_t k, Weighting weighting);

    /**
     * @brief Предсказывает класс для каждого объекта.
     * @param features Матрица признаков.
     * @return Вектор числовых кодов классов.
     * @throw NotFitted Если fit не вызывался.
     * @throw FeatureCountMismatch Если число признаков не совпадает с train.
     */
    std::vector<double> predict(const Matrix& features) const override;
};

#endif