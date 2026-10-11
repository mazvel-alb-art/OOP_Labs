#ifndef MANHATTANDISTANCE_HPP
#define MANHATTANDISTANCE_HPP

#include "IDistance.hpp"

// Манхэттенская метрика (L1): Σ|a_i - b_i|.
class ManhattanDistance : public IDistance
{
    public:
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};
#endif