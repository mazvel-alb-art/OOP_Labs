#ifndef DATASET_HPP
#define DATASET_HPP

#include "Column.hpp"
#include "CategoricalColumn.hpp"
#include "NumericColumn.hpp"
#include "FeatureInfo.hpp"
#include "Matrix.hpp"
#include <vector>
#include <variant>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>

/**
 * @brief Результат разбиения датасета на train/test.
 */
struct Split {
    Matrix train_X;
    Matrix test_X;
    std::vector<double> train_y;
    std::vector<double> test_y;
};

/**
 * @brief Табличный датасет: столбцы, статистика, предобработка.
 *
 * Хранит столбцы двух типов (числовые и категориальные) в std::variant.
 * Пропуски — std::nullopt. По запросу анализирует (analyze) и
 * преобразует данные в плотную числовую матрицу (preprocess).
 */
class Dataset
{
    private:
    bool valid_ = false;
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> column_order;
    std::size_t row_count_ = 0;
    std::size_t NUM_BINS = 8;

    std::string target_name_ = "A16";
    Matrix matrix_;

    public:
    /**
     * @brief Число объектов в датасете.
     */
    std::size_t get_row_count() const { return row_count_; }

    /**
     * @brief Загружает датасет из CSV-файла.
     * @param file_way Путь к файлу.
     *
     * Определяет тип каждого столбца автоматически. Пропуски (?)
     * распознаются и хранятся как std::nullopt.
     */
    Dataset(const std::string file_way);

    /**
     * @brief Печатает сводку: объекты, признаки, пропуски.
     */
    void print_summary();

    /**
     * @brief Печатает статистику по признаку x.
     * @param x Номер признака.
     */
    void statistics(int x);

    /**
     * @brief Считает статистику по всем признакам.
     * @return Мапа: имя признака -> FeatureInfo.
     */
    DatasetInfo analyze();

    /**
     * @brief Проверяет согласованность CSV.
     * @return true если все строки имеют одинаковое число полей.
     */
    bool consistency_check(std::ifstream& file, std::string line1, std::string buffer, int count);

    std::size_t null_count(NumericColumn* nume);
    void min_max_mean(NumericColumn* nume, double& min, double& max, double& mean, int& null_count);
    void variance(NumericColumn* nume, double& var, double& mean, int& null_count);
    std::vector<double> q(NumericColumn* nume, const std::vector<double>& qs);
    void intervals(NumericColumn* nume, std::vector<std::size_t>& bins);

    std::size_t null_count(CategoricalColumn* cate);
    void category_list_periodicity(CategoricalColumn* cate, std::vector<std::string>& category_list, std::vector<int>& periodicity);

    std::vector<double> q50();

    /**
     * @brief Предобработка датасета.
     * @return Ссылка на матрицу признаков.
     *
     * Заполняет пропуски (медиана для чисел, __MISSING__ для категорий),
     * масштабирует числовые признаки (min-max), кодирует категориальные
     * (one-hot). Столбец отклика (target_name_) НЕ попадает в матрицу.
     */
    Matrix& preprocess();

    /**
     * @brief Имя столбца отклика.
     */
    const std::string& get_target_name() const { return target_name_; }

    /**
     * @brief Доступ к столбцу по индексу.
     * @throw std::out_of_range Если индекс вне диапазона.
     */
    const Column& get_columns(int x) const;

    /**
     * @brief Число признаков в датасете.
     */
    std::size_t get_column_count() const {return column_order.size();};

    /**
     * @brief Квантиль по отсортированному массиву (линейная интерполяция).
     * @throw std::invalid_argument Если массив пуст или q вне [0,1].
     */
    static double quantile(const std::vector<double>& sorted, double q);

    /**
     * @brief Число пропусков в признаке x.
     */
    std::size_t get_missing_count(int x) const;

    /**
     * @brief Флаг валидности датасета (CSV согласован).
     */
    bool is_valid() const { return valid_; }

    /**
     * @brief Кодирует столбец отклика в числа: "+" -> 1.0, "-" -> 0.0.
     */
    std::vector<double> get_target_encoded() const;

    /**
     * @brief Делит данные на train/test детерминированно.
     * @param ratio Доля train (0 < ratio < 1).
     * @return Структура Split с train_X, train_y, test_X, test_y.
     * @throw std::invalid_argument Если ratio вне (0,1) или разбиение даёт пустую часть.
     */
    Split train_test_split(double ratio) const;
};

#endif