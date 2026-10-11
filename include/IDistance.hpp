#ifndef IDISTANCE_HPP
#define IDISTANCE_HPP

#include <span>

// Абстрактный интерфейс метрики расстояния.
// Любая метрика обязана уметь вычислить расстояние между двумя векторами.
class IDistance
{
    public:
    virtual ~IDistance() = default;

    // Считает расстояние между векторами a и b.
    // Требование: a.size() == b.size(), иначе бросается исключение.
    virtual double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const = 0;
};

#endif