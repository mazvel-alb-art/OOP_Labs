#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <span>

/**
 * @brief Плотная числовая матрица с плоским хранением данных (RowMajor).
 *
 * Данные лежат в одном непрерывном std::vector<double>. Строка i —
 * это подотрезок [i*cols_, (i+1)*cols_). Это позволяет возвращать
 * std::span на строку без копирования.
 */
class Matrix
{
private:
    std::vector<double> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<std::string> matrix_column_names;

public:
    Matrix() = default;

    /**
     * @brief Число строк матрицы.
     */
    std::size_t rows() const { return rows_; }

    /**
     * @brief Число столбцов матрицы.
     */
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

    /**
     * @brief Доступ к внутреннему массиву данных (для записи).
     */
    std::vector<double>&       data()       { return data_; }

    /**
     * @brief Доступ к внутреннему массиву данных (для чтения).
     */
    const std::vector<double>& data() const { return data_; }

    /**
     * @brief Имена столбцов (для one-hot — составные, например A4_u).
     */
    std::vector<std::string>&       get_matrix_column_names()       { return matrix_column_names; }

    /**
     * @brief Имена столбцов (только чтение).
     */
    const std::vector<std::string>& get_matrix_column_names() const { return matrix_column_names; }

    /**
     * @brief Сохраняет матрицу в бинарный файл.
     * @param file_name Путь к файлу.
     *
     * Формат: [rows:size_t][cols:size_t][data:double * rows*cols].
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
     * @return std::span на строку длины cols_.
     *
     * Границы не проверяет. Для проверяемого доступа — at().
     */
    std::span<const double> operator[](std::size_t i) const;

    /**
     * @brief Доступ к строке i без копирования (чтение и запись).
     * @param i Индекс строки.
     * @return std::span на строку длины cols_.
     */
    std::span<double> operator[](std::size_t i);

    /**
     * @brief Проверяемый доступ к элементу (i, j).
     * @param i Индекс строки.
     * @param j Индекс столбца.
     * @return Ссылка на элемент.
     * @throw std::out_of_range Если i или j вне границ.
     */
    double& at(std::size_t i, std::size_t j);

    /**
     * @brief Проверяемый доступ к элементу (i, j), только чтение.
     */
    const double& at(std::size_t i, std::size_t j) const;
};

#endif