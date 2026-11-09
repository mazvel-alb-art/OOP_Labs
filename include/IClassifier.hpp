#ifndef ICLASSIFIER_HPP
#define ICLASSIFIER_HPP

#include "IModel.hpp"

/**
 * @brief Общий контракт классификаторов.
 *
 * Добавляет методы для получения матрицы вероятностей и списка классов.
 */
class IClassifier : public IModel {
public:
    virtual ~IClassifier() = default;

    /**
     * @brief Предсказывает вероятности классов для объектов.
     * @param X Матрица признаков.
     * @return Матрица размера X.rows() × classes().size().
     *         Столбец c — вероятность класса classes()[c].
     * @throw NotFitted Если модель не обучена.
     * @throw FeatureCountMismatch Если число признаков не совпадает с обучающим.
     */
    virtual Matrix predict_proba(const Matrix& X) const = 0;

    /**
     * @brief Возвращает список классов, которые выучила модель.
     *        Классы отсортированы по возрастанию исходных меток.
     * @throw NotFitted Если модель не обучена.
     */
    virtual std::span<const double> classes() const = 0;
};

#endif