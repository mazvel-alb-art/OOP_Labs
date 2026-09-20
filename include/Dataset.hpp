#ifndef DATASET_HPP
#define DATASET_HPP

#include "Column.hpp"
#include "CategoricalColumn.hpp"
#include "NumericColumn.hpp"

#include <variant> 
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>


class Dataset
{

    private:
    std::unordered_map<std::string, Column> columns;
    public:
    Dataset(const std::string file_way);
    void print_summary();


};

#endif