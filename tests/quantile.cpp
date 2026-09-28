#include "Dataset.hpp"
#include <iostream>
#include <vector>
#include <cmath>

bool approx_equal(double a, double b, double eps = 1e-9){
    return std::fabs(a - b) < eps;
}

int main(){
    // Тест 1: нечётное число элементов
    // {1, 2, 3, 4, 5}
    //   q05 = 1.2, q25 = 2, q50 = 3, q75 = 4, q95 = 4.8
    {
        std::vector<double> data = {1, 2, 3, 4, 5};
        if (!approx_equal(Dataset::quantile(data, 0.05), 1.2))  { std::cout << "Quantile: q05 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.25), 2.0))  { std::cout << "Quantile: q25 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.50), 3.0))  { std::cout << "Quantile: q50 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.75), 4.0))  { std::cout << "Quantile: q75 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.95), 4.8))  { std::cout << "Quantile: q95 fail\n"; return 1; }
    }

    // Тест 2: чётное число элементов
    // {1, 2, 3, 4}
    //   q25 = 1.75, q50 = 2.5, q75 = 3.25
    {
        std::vector<double> data = {1, 2, 3, 4};
        if (!approx_equal(Dataset::quantile(data, 0.25), 1.75)) { std::cout << "Quantile: even q25 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.50), 2.5))  { std::cout << "Quantile: even q50 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.75), 3.25)) { std::cout << "Quantile: even q75 fail\n"; return 1; }
    }

    // Тест 3: один элемент
    {
        std::vector<double> data = {42};
        if (!approx_equal(Dataset::quantile(data, 0.0), 42)) { std::cout << "Quantile: single q0 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 1.0), 42)) { std::cout << "Quantile: single q1 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 0.5), 42)) { std::cout << "Quantile: single q50 fail\n"; return 1; }
    }

    // Тест 4: крайние квантили
    {
        std::vector<double> data = {10, 20, 30};
        if (!approx_equal(Dataset::quantile(data, 0.0), 10)) { std::cout << "Quantile: q0 fail\n"; return 1; }
        if (!approx_equal(Dataset::quantile(data, 1.0), 30)) { std::cout << "Quantile: q1 fail\n"; return 1; }
    }

    std::cout << "Quantile: Тест пройден: квантили считаются верно\n";
    return 0;
}