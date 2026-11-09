#include "Matrix.hpp"
#include "Exceptions.hpp"
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
std::span<const double> Matrix::operator[](std::size_t i) const {
    if (layout_ == MatrixLayout::ColumnMajor && cols_ > 1) {
        throw NonContiguousSliceError("Matrix::operator[]: неплотная строка в ColumnMajor");
    }
    return std::span<const double>(data_.data() + i * cols_, cols_);
}

std::span<double> Matrix::operator[](std::size_t i){
     if (layout_ == MatrixLayout::ColumnMajor && cols_ > 1) {
        throw NonContiguousSliceError("Matrix::operator[]: неплотная строка в ColumnMajor");
    }
    return std::span<double>(data_.data() + i * cols_, cols_);
}

double& Matrix::at(std::size_t i, std::size_t j){
    if (i >= rows_ || j >= cols_){
        throw std::out_of_range("Matrix::at: индекс вне границ");
    }
    if (layout_ == MatrixLayout::RowMajor){
        return data_[i * cols_ + j];
    }
    else{
        return data_[j * rows_ + i];
    }
}

const double& Matrix::at(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_){
        throw std::out_of_range("Matrix::at: индекс вне границ");
    }
    if (layout_ == MatrixLayout::RowMajor){
        return data_[i * cols_ + j];
    }
    else{
        return data_[j * rows_ + i];
    }
}
//3
double& Matrix::operator()(std::size_t row, std::size_t col) &{
    return at(row, col);
}
const double& Matrix::operator()(std::size_t row, std::size_t col) const &{
    return at(row, col);
}
//
MatrixSlice<double> Matrix::operator()(std::size_t row, std::nullopt_t) &{
    if (row >= rows_){
        throw std::out_of_range("Matrix::operator(): row out of range");
    }
    if (cols_ == 0 ){
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Row);
    }
    if (layout_ == MatrixLayout::RowMajor){
        return MatrixSlice<double>(data_.data() + row * cols_, cols_, 1, SliceAxis::Row);
    }
    else{
        return MatrixSlice<double>(data_.data() + row, cols_, rows_, SliceAxis::Row);
    }
}
MatrixSlice<const double> Matrix::operator()(std::size_t row, std::nullopt_t) const &{
    if (row >= rows_){
        throw std::out_of_range("Matrix::operator(): row out of range");
    }
    if (cols_ == 0 ){
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Row);
    }
    if (layout_ == MatrixLayout::RowMajor){
        return MatrixSlice<const double>(data_.data() + row * cols_, cols_, 1, SliceAxis::Row);
    }
    else{
        return MatrixSlice<const double>(data_.data() + row, cols_, rows_, SliceAxis::Row);
    }
}
//
MatrixSlice<double> Matrix::operator()(std::nullopt_t, std::size_t col) &{
    if (col >= cols_){
        throw std::out_of_range("Matrix::operator(): col out of range");
    }
    if (rows_ == 0 ){
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Column);
    }
    if (layout_ == MatrixLayout::ColumnMajor){
        return MatrixSlice<double>(data_.data() + col * rows_, rows_, 1, SliceAxis::Column);
    }
    else{
        return MatrixSlice<double>(data_.data() + col, rows_, cols_, SliceAxis::Column);
    }
}
MatrixSlice<const double> Matrix::operator()(std::nullopt_t, std::size_t col) const &{
    if (col >= cols_){
        throw std::out_of_range("Matrix::operator(): col out of range");
    }
    if (rows_ == 0 ){
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Column);
    }
    if (layout_ == MatrixLayout::ColumnMajor){
        return MatrixSlice<const double>(data_.data() + col * rows_, rows_, 1, SliceAxis::Column);
    }
    else{
        return MatrixSlice<const double>(data_.data() + col, rows_, cols_, SliceAxis::Column);
    }
}