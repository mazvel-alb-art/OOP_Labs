#ifndef EUCLIDEANDISTANCE_HPP
#define EUCLIDEANDISTANCE_HPP

#include "IDistance.hpp"

// Евклидова метрика (L2): sqrt(Σ(a_i - b_i)²).
class EuclideanDistance : public IDistance
{
    public:
    double compute(
        std::span<const double> a,
        std::span<const double> b
    ) const override;
};

#endif