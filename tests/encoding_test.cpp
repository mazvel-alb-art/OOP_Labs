#include "Dataset.hpp"
#include <iostream>
#include <variant>
#include <string>
#include <vector>
#include <cmath>

int main(){
    Dataset dataset("data/crx.csv");
    dataset.preprocess();

    const auto& matrix = dataset.get_matrix();
    const auto& names = dataset.get_matrix_column_names();

    // 1) Матрица непустая
    if (matrix.empty()){
        std::cout << "One_hot: Матрица пуста" << std::endl;
        return 1;
    }

    // 2) Число столбцов совпадает с числом имён
    std::size_t cols = matrix[0].size();
    if (names.size() != cols){
        std::cout << "One_hot: names.size() = " << names.size()<< ", matrix[0].size() = " << cols << std::endl;
        return 1;
    }

    // 3) Число столбцов больше числа исходных признаков (после one-hot)
    std::size_t original_count = dataset.get_column_count();
    if (cols <= original_count){
        std::cout << "One_hot: столбцов " << cols<< " <= исходных признаков " << original_count<< " (ожидалось больше)" << std::endl;
        return 1;
    }

    // 4) Есть составные имена вида "A4_u" (хотя бы одно с '_')
    bool has_composite = false;
    for (std::size_t k = 0; k < names.size(); k++){
        if (names[k].find('_') != std::string::npos){
            has_composite = true;
            break;
        }
    }
    if (!has_composite){
        std::cout << "One_hot: не найдено ни одного составного имени вида A4_u" << std::endl;
        return 1;
    }

    // 5) Все значения матрицы лежат в [0, 1]
    //    (min-max для числовых даёт [0,1], one-hot — 0 или 1)
    for (std::size_t j = 0; j < matrix.size(); j++){
        for (std::size_t k = 0; k < matrix[j].size(); k++){
            double v = matrix[j][k];
            if (v < 0.0 || v > 1.0){
                std::cout << "One_hot: matrix[" << j << "][" << k
                          << "] = " << v << " вне [0, 1]" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "One_hot: Тест пройден, столбцов " << cols
              << " (исходных " << original_count << ")" << std::endl;
    return 0;
}