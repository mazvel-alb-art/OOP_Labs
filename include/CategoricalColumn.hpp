#ifndef CATEGORICALCOLUMN_HPP
#define CATEGORICALCOLUMN_HPP

#include <vector>
#include <optional>
#include <variant>
#include <string>


struct CategoricalColumn
{
    std::vector<std::optional<std::string>> values;
};

#endif