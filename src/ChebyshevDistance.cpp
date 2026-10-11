#include "ChebyshevDistance.hpp"

#include <cmath>
#include <stdexcept>

double ChebyshevDistance::compute(std::span<const double> a, std::span<const double> b) const{
    if (a.size() != b.size()){
        throw std::invalid_argument("ChebyshevDistance: размерности не совпадают");
    }
    double max = std::abs(a[0] - b[0]);
    for (std::size_t i = 0; i < a.size();i++){
        double d = std::abs(a[i] - b[i]);
        if (d > max){
            max = d;
        }
    }
    return max;
}