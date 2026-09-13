#ifndef NUMERICCOLUMN_HPP
#define NUMERICCOLUMN_HPP


#include <vector>
#include <optional>

struct NumericColumn
{
    std::vector<std::optional<double>> values;
};

#endif