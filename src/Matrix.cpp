#include "Matrix.hpp"
#include <fstream>
#include <iostream>
#include <span>

//1
void Matrix::save_binary(const std::string& file_name) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для записи: " << file_name << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    file.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    file.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(double));
}

void Matrix::load_binary(const std::string& file_name){
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для чтения: " << file_name << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(&rows_), sizeof(rows_));
    file.read(reinterpret_cast<char*>(&cols_), sizeof(cols_));
    data_.resize(rows_ * cols_);
    file.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(double));
}
//2
std::span<const double> Matrix::operator[](std::size_t i) const{
    return std::span<const double>(data_.data() + i * cols_, cols_);
}

std::span<double> Matrix::operator[](std::size_t i){
    return std::span<double>(data_.data() + i * cols_, cols_);
}

double& Matrix::at(std::size_t i, std::size_t j){
    if (i >= rows_ || j >= cols_){
        throw std::out_of_range("Matrix::at: индекс вне границ");
    }
    return data_[i * cols_ + j];
}

const double& Matrix::at(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_){
        throw std::out_of_range("Matrix::at: индекс вне границ");
    }
    return data_[i * cols_ + j];
}