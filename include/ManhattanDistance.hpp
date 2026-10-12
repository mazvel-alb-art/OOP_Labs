#ifndef MANHATTANDISTANCE_HPP
#define MANHATTANDISTANCE_HPP

#include "IDistance.hpp"

/**
 * @brief Манхэттенская метрика (L1): Σ|a_i - b_i|.
 */
class ManhattanDistance : public IDistance
{
    public:
    /**
     * @brief Считает манхэттенское расстояние между двумя векторами.
     * @param a Первый вектор.
     * @param b Второй вектор.
     * @return Σ|a_i - b_i|.
     * @throw DimensionMismatch Если a.size() != b.size().
     */
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};
#endif