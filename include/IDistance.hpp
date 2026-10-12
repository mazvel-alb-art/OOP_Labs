#ifndef IDISTANCE_HPP
#define IDISTANCE_HPP

#include <span>

/**
 * @brief Абстрактный интерфейс метрики расстояния.
 *
 * Метрика — самостоятельная сущность. Модель kNN принимает её извне
 * и не знает, какая именно реализация за указателем. Добавление новой
 * метрики не требует изменений в коде модели.
 */
class IDistance
{
    public:
    virtual ~IDistance() = default;

    /**
     * @brief Считает расстояние между двумя векторами.
     * @param a Первый вектор.
     * @param b Второй вектор.
     * @return Расстояние.
     * @throw DimensionMismatch Если a.size() != b.size().
     */
    virtual double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const = 0;
};

#endif