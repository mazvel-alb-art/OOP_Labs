#ifndef EUCLIDEANDISTANCE_HPP
#define EUCLIDEANDISTANCE_HPP

#include "IDistance.hpp"

/**
 * @brief Евклидова метрика (L2): sqrt(Σ(a_i - b_i)²).
 */
class EuclideanDistance : public IDistance
{
    public:
    /**
     * @brief Считает евклидово расстояние между двумя векторами.
     * @param a Первый вектор.
     * @param b Второй вектор.
     * @return sqrt(Σ(a_i - b_i)²).
     * @throw DimensionMismatch Если a.size() != b.size().
     */
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};

#endif