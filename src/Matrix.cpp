#include "Matrix.hpp"
#include <fstream>
#include <iostream>

void Matrix::save_binary(const std::string& file_name)  {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для записи: " << file_name << std::endl;
        return;
    }
    std::size_t rows = matrix.size();
    std::size_t columns1 = matrix[0].size();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns1), sizeof(columns1));
    for (std::size_t i = 0; i < rows; i++){
        file.write(reinterpret_cast<const char*>(matrix[i].data()), columns1 * sizeof(double));
    }
}

void Matrix::load_binary(const std::string& file_name){
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для чтения: " << file_name << std::endl;
        return;
    }
    std::size_t rows;
    std::size_t columns1;

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns1), sizeof(columns1));
    matrix.resize(rows);
    for (std::size_t i = 0; i < rows; i++){
        matrix[i].resize(columns1);
        file.read(reinterpret_cast<char*>(matrix[i].data()), columns1 * sizeof(double));
    }
}