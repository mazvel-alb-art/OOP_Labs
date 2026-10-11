#include "ManhattanDistance.hpp"

#include <stdexcept>
#include <cmath>

double ManhattanDistance::compute(std::span<const double> a, std::span<const double> b) const{
    if (a.size() != b.size()){
        throw std::invalid_argument("ManhattanDistance: размерности не совпадают");
    }

    double sum =0.0;
    for (std::size_t i = 0; i < a.size();i++){
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}