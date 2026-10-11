#include "EuclideanDistance.hpp"
#include <cmath>
#include <stdexcept>

double EuclideanDistance::compute(
    std::span<const double> a,
    std::span<const double> b
) const
{
    if (a.size() != b.size()){
        throw std::invalid_argument("EuclideanDistance: размерности не совпадают");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); i++){
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}