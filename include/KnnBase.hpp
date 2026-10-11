#ifndef KNNBASE_HPP
#define KNNBASE_HPP

#include "IDistance.hpp"
#include "Matrix.hpp"

#include <memory>
#include <span>
#include <vector>
#include <utility>
#include <cstddef>
class KnnBase
{
    protected:
    std::unique_ptr<IDistance> metric_;
    std::size_t k_;
    const Matrix* train_features_ = nullptr;
    std::span<const double> train_target_;

    public:
    KnnBase(std::unique_ptr<IDistance> metric, std::size_t k);
    virtual ~KnnBase() = default;

    void fit(const Matrix& features, std::span<const double> target);
    std::vector<std::pair<double, std::size_t>> find_neighbors(std::span<const double> query,std::size_t k) const;
};

 #endif