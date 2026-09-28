#ifndef DATASET_HPP
#define DATASET_HPP

#include "Column.hpp"
#include "CategoricalColumn.hpp"
#include "NumericColumn.hpp"
#include "Statistics.hpp"
#include <vector>
#include <variant> 
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>


class Dataset
{

    private:
    bool valid_ = false;
    std::unordered_map<std::string, Column> columns;
    std::vector<std::string> column_order;

    std::vector<std::string> matrix_columns;

    Statistic statistic;

    std::vector<std::vector<double>> matrix;
    std::vector<std::string> matrix_column_names;
    public:
    Dataset(const std::string file_way);
    void print_summary();
    void statistics(int x);

    bool consistency_check(std::ifstream& file, std::string line1, std::string buffer, int count);

    std::size_t null_count(NumericColumn* nume);
    void min_max_mean(NumericColumn* nume, double& min, double& max, double& mean, int& null_count);
    void var(NumericColumn* nume, double& var, double& mean, int& null_count);
    void q(NumericColumn* nume, double& q05, double& q25, double& q50, double& q75, double& q95);
    void intervals(NumericColumn* nume, int& interval1, int& interval2, int& interval3, int& interval4);

    std::size_t null_count(CategoricalColumn* cate);
    void category_list_periodicity(CategoricalColumn* cate, std::vector<std::string>& category_list, std::vector<int>& periodicity);

    void q50();

    void preprocess();

    void save_binary(const std::string& file_name);
    void load_binary(const std::string& file_name);
    //tests
    const std::vector<std::vector<double>>& get_matrix() const;
    const Column& get_columns(int x) const;
    std::size_t get_column_count() const {return column_order.size();};
    static double quantile(const std::vector<double>& sorted, double q);
    std::size_t get_missing_count(int x) const;
    const std::vector<std::string>& get_matrix_column_names() const;
    bool is_valid() const { return valid_; }

};

#endif