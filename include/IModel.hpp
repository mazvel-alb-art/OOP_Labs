#ifndef IMODEL_HPP
#define IMODEL_HPP

#include "Matrix.hpp"

/**
 * @brief Базовый интерфейс модели машинного обучения.
 *
 * Все модели курса (kNN в работе 2, ядерная регрессия в работе 3,
 * композиции в работе 4) реализуют этот интерфейс. Контракт общий:
 * сначала fit — обучение, потом predict — предсказание.
 */
class IModel {
public:
    virtual ~IModel() = default;

    /**
     * @brief Обучает модель на обучающей выборке.
     * @param features Матрица признаков (объекты — строки).
     * @param target Вектор откликов, target[i] соответствует features[i].
     * @throw EmptyTrainingSet Если выборка пустая.
     * @throw DimensionMismatch Если features.rows() != target.size().
     */
    virtual void fit(const Matrix& features, std::span<const double> target) = 0;

    /**
     * @brief Предсказывает отклики для объектов.
     * @param features Матрица признаков (объекты — строки).
     * @return Вектор предсказаний той же длины, что features.rows().
     * @throw NotFitted Если fit ещё не вызывался.
     * @throw FeatureCountMismatch Если число признаков не совпадает с обучающим.
     */
    virtual std::vector<double> predict(const Matrix& features) const = 0;
};

#endif