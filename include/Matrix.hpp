#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "MatrixSlice.hpp"

#include <vector>
#include <string>
#include <cstddef>
#include <span>
#include <optional>

/**
 * @brief Порядок хранения элементов матрицы.
 */
enum class MatrixLayout { RowMajor, ColumnMajor };

/**
 * @brief Плотная числовая матрица с плоским хранением данных.
 *
 * Данные лежат в одном непрерывном `std::vector<double>`. Поддерживаются
 * оба порядка хранения (RowMajor и ColumnMajor). Через `operator()` можно
 * получать как отдельные элементы, так и срезы строк/столбцов.
 */
class Matrix
{
private:
    std::vector<double> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<std::string> matrix_column_names;
    MatrixLayout layout_ = MatrixLayout::RowMajor;

public:
    Matrix() = default;

    /** @brief Число строк матрицы. */
    std::size_t rows() const { return rows_; }

    /** @brief Число столбцов матрицы. */
    std::size_t cols() const { return cols_; }

    /**
     * @brief Устанавливает размеры матрицы.
     * @param rows Число строк.
     * @param cols Число столбцов.
     *
     * Внимание: не изменяет содержимое data_. Используется для указания
     * размеров до заполнения.
     */
    void set_size(std::size_t rows, std::size_t cols) {
        rows_ = rows;
        cols_ = cols;
    }

    /** @brief Доступ к внутреннему массиву данных (для записи). */
    std::vector<double>&       data()       { return data_; }

    /** @brief Доступ к внутреннему массиву данных (для чтения). */
    const std::vector<double>& data() const { return data_; }

    /** @brief Имена столбцов (для записи). */
    std::vector<std::string>&       get_matrix_column_names()       { return matrix_column_names; }

    /** @brief Имена столбцов (только чтение). */
    const std::vector<std::string>& get_matrix_column_names() const { return matrix_column_names; }

    /**
     * @brief Сохраняет матрицу в бинарный файл.
     * @param file_name Путь к файлу.
     */
    void save_binary(const std::string& file_name);

    /**
     * @brief Загружает матрицу из бинарного файла.
     * @param file_name Путь к файлу.
     */
    void load_binary(const std::string& file_name);

    /**
     * @brief Доступ к строке i без копирования (только чтение).
     * @param i Индекс строки.
     * @throw NonContiguousSliceError Если строка неплотная (ColumnMajor и cols > 1).
     */
    std::span<const double> operator[](std::size_t i) const;

    /**
     * @brief Доступ к строке i без копирования (чтение и запись).
     * @param i Индекс строки.
     * @throw NonContiguousSliceError Если строка неплотная (ColumnMajor и cols > 1).
     */
    std::span<double> operator[](std::size_t i);

    /**
     * @brief Проверяемый доступ к элементу (i, j).
     * @throw std::out_of_range Если i или j вне границ.
     */
    double& at(std::size_t i, std::size_t j);

    /**
     * @brief Проверяемый доступ к элементу (i, j), только чтение.
     * @throw std::out_of_range Если i или j вне границ.
     */
    const double& at(std::size_t i, std::size_t j) const;

    /** @brief Установить порядок хранения. */
    void set_layout(MatrixLayout layout) { layout_ = layout; }

    /** @brief Текущий порядок хранения. */
    MatrixLayout layout() const { return layout_; }

    /**
     * @brief Доступ к элементу (row, col) по ссылке (lvalue).
     */
    double& operator()(std::size_t row, std::size_t col) &;
    /**
     * @brief Доступ к элементу (row, col) по константной ссылке (lvalue).
     */
    const double& operator()(std::size_t row, std::size_t col) const &;

    /** @brief Запрет доступа к элементу у временной матрицы. */
    double& operator()(std::size_t row, std::size_t col) && = delete;
    /** @brief Запрет доступа к элементу у временной константной матрицы. */
    const double& operator()(std::size_t row, std::size_t col) const && = delete;

    /**
     * @brief Срез строки (lvalue).
     */
    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) &;
    /**
     * @brief Срез строки (константный lvalue).
     */
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const &;

    /** @brief Запрет среза строки у временной матрицы. */
    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) && = delete;
    /** @brief Запрет среза строки у временной константной матрицы. */
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const && = delete;

    /**
     * @brief Срез столбца (lvalue).
     */
    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) &;
    /**
     * @brief Срез столбца (константный lvalue).
     */
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const &;

    /** @brief Запрет среза столбца у временной матрицы. */
    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) && = delete;
    /** @brief Запрет среза столбца у временной константной матрицы. */
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const && = delete;
};

#endif