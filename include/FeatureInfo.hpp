#ifndef NUMERICFEATUREINFO_HPP
#define NUMERICFEATUREINFO_HPP

#include <vector>
#include <cstddef>
#include <unordered_map>
#include <variant>
#include <string>

struct NumericFeatureInfo
{
    std::size_t missing_count = 0;

    double min = 0.0;
    double max = 0.0;

    double mean = 0.0;
    double var = 0.0;

    double median = 0.0;

    double q05 = 0.0;
    double q25 = 0.0;
    double q75 = 0.0;
    double q95 = 0.0;
};

struct CategoricalFeatureInfo
{
    std::size_t missing_count = 0;

    std::vector<std::string> categories;

    std::unordered_map<std::string, std::size_t> frequencies;
};

using FeatureInfo = std::variant<NumericFeatureInfo, CategoricalFeatureInfo>;

using DatasetInfo = std::unordered_map<std::string, FeatureInfo>;

#endif