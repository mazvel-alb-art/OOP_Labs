#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>
#include <cstddef>

class Matrix
{
    private:
    std::vector<std::vector<double>> matrix;
    std::vector<std::string> matrix_column_names;

    public:
    Matrix() = default;

    std::vector<std::vector<double>>& get_matrix() { return matrix; }

    std::vector<std::string>& get_matrix_column_names() { return matrix_column_names; }

    void save_binary(const std::string& file_name);
    void load_binary(const std::string& file_name);
};

#endif