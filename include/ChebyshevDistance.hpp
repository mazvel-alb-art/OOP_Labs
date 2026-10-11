#ifndef CHEBYSHEVDISTANCE_HPP
#define CHEBYSHEVDISTANCE_HPP

#include "IDistance.hpp"

// Метрика Чебышёва (L∞): max|a_i - b_i|.
class ChebyshevDistance : public IDistance
{
    public:
    double compute(std::span<const double>, std::span<const double>) const override;
};

#endif