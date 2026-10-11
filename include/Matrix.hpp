#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <span>


class Matrix
{
private:
    std::vector<double> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<std::string> matrix_column_names;

public:
    //1
    Matrix() = default;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    void set_size(std::size_t rows, std::size_t cols) {
        rows_ = rows;
        cols_ = cols;
    }

    std::vector<double>&       data()       { return data_; }
    const std::vector<double>& data() const { return data_; }

    std::vector<std::string>&       get_matrix_column_names()       { return matrix_column_names; }
    const std::vector<std::string>& get_matrix_column_names() const { return matrix_column_names; }

    void save_binary(const std::string& file_name);
    void load_binary(const std::string& file_name);
    //2
    std::span<const double> operator[](std::size_t i) const;
    std::span<double> operator[](std::size_t i); 
    double& at(std::size_t i, std::size_t j);
    const double& at(std::size_t i, std::size_t j) const;
};

#endif