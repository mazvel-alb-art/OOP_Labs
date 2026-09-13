#ifndef DATASET_HPP
#define DATASET_HPP

#include "Column.hpp"
#include "CategoricalColumn.hpp"
#include "NumericColumn.hpp"

#include <variant> 

class Dataset
{
    private:
    using Column = std::variant<NumericColumn, CategoricalColumn>;
};

#endif