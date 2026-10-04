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


// todo: предобработанная матрица - отдельная сущность (класс)
// добавить отдельные хеш мапы со статистиками
// сделать изменяемое количество бинов в гистограмме 
class Dataset
{

    private:
    bool valid_ = false;
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> column_order;
    std::size_t row_count_ = 0;
    std::size_t NUM_BINS = 8;


    Matrix matrix_;
    public:
    std::size_t get_row_count() const { return row_count_; }

    Dataset(const std::string file_way);
    void print_summary();
    void statistics(int x);

    DatasetInfo analyze();

    bool consistency_check(std::ifstream& file, std::string line1, std::string buffer, int count);

    std::size_t null_count(NumericColumn* nume);
    void min_max_mean(NumericColumn* nume, double& min, double& max, double& mean, int& null_count);
    void variance(NumericColumn* nume, double& var, double& mean, int& null_count);
    std::vector<double> q(NumericColumn* nume, const std::vector<double>& qs);
    void intervals(NumericColumn* nume, std::vector<std::size_t>& bins);

    std::size_t null_count(CategoricalColumn* cate);
    void category_list_periodicity(CategoricalColumn* cate, std::vector<std::string>& category_list, std::vector<int>& periodicity);

    std::vector<double> q50();

    Matrix& preprocess(); 


    //tests
    const Column& get_columns(int x) const;
    std::size_t get_column_count() const {return column_order.size();};
    static double quantile(const std::vector<double>& sorted, double q);
    std::size_t get_missing_count(int x) const;
    bool is_valid() const { return valid_; }

};

#endif