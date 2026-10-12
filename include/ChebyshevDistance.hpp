#ifndef CHEBYSHEVDISTANCE_HPP
#define CHEBYSHEVDISTANCE_HPP

#include "IDistance.hpp"

/**
 * @brief Метрика Чебышёва (L∞): max|a_i - b_i|.
 */
class ChebyshevDistance : public IDistance
{
    public:
    /**
     * @brief Считает расстояние Чебышёва между двумя векторами.
     * @param a Первый вектор.
     * @param b Второй вектор.
     * @return max|a_i - b_i|.
     * @throw DimensionMismatch Если a.size() != b.size().
     */
    double compute(std::span<const double>, std::span<const double>) const override;
};

#endif