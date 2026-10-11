#include "EuclideanDistance.hpp"
#include "ManhattanDistance.hpp"
#include "ChebyshevDistance.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

bool approx_equal(double a, double b, double eps = 1e-9){
    return std::fabs(a - b) < eps;
}

int main(){
    // a = (0, 0), b = (3, 4)
    //   Euclid:    sqrt(9 + 16) = 5
    //   Manhattan: 3 + 4 = 7
    //   Chebyshev: max(3, 4) = 4
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};

    EuclideanDistance euclid;
    ManhattanDistance manhattan;
    ChebyshevDistance chebyshev;

    if (!approx_equal(euclid.compute(a, b), 5.0)){
        std::cout << "Distance: Euclidean fail\n";
        return 1;
    }
    if (!approx_equal(manhattan.compute(a, b), 7.0)){
        std::cout << "Distance: Manhattan fail\n";
        return 1;
    }
    if (!approx_equal(chebyshev.compute(a, b), 4.0)){
        std::cout << "Distance: Chebyshev fail\n";
        return 1;
    }

    // Проверка размерностей — все три метрики должны бросить исключение
    std::vector<double> short_vec = {1.0, 2.0};
    std::vector<double> long_vec  = {1.0, 2.0, 3.0};

    try {
        euclid.compute(short_vec, long_vec);
        std::cout << "Distance: Euclidean не бросила исключение\n";
        return 1;
    }
    catch (const std::invalid_argument&){ /* OK */ }

    try {
        manhattan.compute(short_vec, long_vec);
        std::cout << "Distance: Manhattan не бросила исключение\n";
        return 1;
    }
    catch (const std::invalid_argument&){ /* OK */ }

    try {
        chebyshev.compute(short_vec, long_vec);
        std::cout << "Distance: Chebyshev не бросила исключение\n";
        return 1;
    }
    catch (const std::invalid_argument&){ /* OK */ }

    // Через интерфейс IDistance — проверим полиморфизм
    IDistance* metric = &euclid;
    if (!approx_equal(metric->compute(a, b), 5.0)){
        std::cout << "Distance: полиморфизм через IDistance fail\n";
        return 1;
    }

    std::cout << "Distance: все тесты пройдены\n";
    return 0;
}