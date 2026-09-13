#ifndef COLUMN_HPP
#define COLUMN_HPP

#include <variant>

#include "CategoricalColumn.hpp"
#include "NumericColumn.hpp"

using Column = std::variant<NumericColumn, CategoricalColumn>;
#endif