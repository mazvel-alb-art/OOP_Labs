#include "KnnBase.hpp"
#include "Exceptions.hpp"
#include <queue>
#include <algorithm>
#include <memory>
#include <utility>
#include <cstddef>

KnnBase::KnnBase(std::unique_ptr<IDistance> metric, std::size_t k)
    : metric_(std::move(metric)), k_(k)
{
    if (k < 1){
        throw InvalidK("KnnBase: k должно быть >= 1");
    }
}
//
void KnnBase::fit(const Matrix& features, std::span<const double> target){
    if (features.rows() == 0){
        throw EmptyTrainingSet("Неверное количество строк!");
    }

    if (features.rows() != target.size()){
        throw DimensionMismatch("Несоответствие размеров");
    }
    train_features_ = &features;
    train_target_ = target;
}
//
std::vector<std::pair<double, std::size_t>> KnnBase::find_neighbors(std::span<const double> query, std::size_t k) const{
    using Pair = std::pair<double, std::size_t>;
    if (train_features_ == nullptr){
        throw NotFitted("Модель не обучена. Отсутствует матрица");
    }

    if ((k == 0) || (k > train_features_->rows())){
        throw InvalidK("Неверное количество соседей");
    }

    if (query.size() != train_features_->cols()){
        throw DimensionMismatch("Размерности не совпали");
    }

    auto cmp = [](const Pair& a, const Pair& b) {       //компоратор. true если a < b. инвертируется в priority_queue. то есть корень(top) - тот для кого все cmp(top, x) == false, а не true
        if (a.first != b.first){
            return (a.first < b.first);
        }
        else{
            return (a.second < b.second);
        }
    };

    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> heap(cmp);

    for (std::size_t i = 0;i < train_features_->rows();i++){
        double distance = metric_->compute(query, (*train_features_)[i]);
        if (heap.size() < k){
            heap.push({distance, i});
        }

        else if (distance < heap.top().first){
            heap.pop();
            heap.push({distance, i});
        }
        else if (distance == heap.top().first){
            if (i < heap.top().second){
                heap.pop();
                heap.push({distance, i});
            }
        }
    }

    std::vector<Pair> result;
    result.reserve(heap.size());
    while (!heap.empty()){
        result.push_back(heap.top());
        heap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}