#include "KnnRegressor.hpp"
#include "EuclideanDistance.hpp"
#include "ManhattanDistance.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <span>

bool approx_equal(double a, double b, double eps = 1e-9){
    return std::fabs(a - b) < eps;
}

// Полная сортировка — только в тесте, для сравнения с кучей.
std::vector<std::pair<double, std::size_t>> full_sort(
    const Matrix& m,
    std::span<const double> query,
    IDistance& metric)
{
    std::vector<std::pair<double, std::size_t>> all;
    all.reserve(m.rows());
    for (std::size_t i = 0; i < m.rows(); i++){
        all.push_back({metric.compute(query, m[i]), i});
    }
    std::sort(all.begin(), all.end());
    return all;
}

int main(){
    // -------- Матрица 5×2 --------
    Matrix m;
    m.set_size(5, 2);
    m.data() = {0, 0,
                1, 1,
                2, 2,
                3, 3,
                4, 4};

    std::vector<double> target = {10, 20, 30, 40, 50};
    std::span<const double> target_span(target);

    // -------- Запрос = строка 0 --------
    std::span<const double> query = m[0];

    // ============================================================
    // ТЕСТ 1: k=3, евклидова метрика — сравнение с полной сортировкой
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        EuclideanDistance ref_metric;

        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);
        knn.fit(m, target_span);

        auto res = knn.find_neighbors(query, 3);
        auto ref = full_sort(m, query, ref_metric);

        if (res.size() != 3){
            std::cout << "Neighbors: размер результата != 3\n";
            return 1;
        }
        for (std::size_t i = 0; i < 3; i++){
            if (res[i].second != ref[i].second){
                std::cout << "Neighbors: индекс " << i << " не совпал: "
                          << res[i].second << " vs " << ref[i].second << "\n";
                return 1;
            }
            if (!approx_equal(res[i].first, ref[i].first)){
                std::cout << "Neighbors: расстояние " << i << " не совпало\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 2: k=2, манхэттенская метрика — сравнение с полной сортировкой
    // ============================================================
    {
        auto metric = std::make_unique<ManhattanDistance>();
        ManhattanDistance ref_metric;

        KnnRegressor knn(std::move(metric), 2, Weighting::Uniform);
        knn.fit(m, target_span);

        auto res = knn.find_neighbors(query, 2);
        auto ref = full_sort(m, query, ref_metric);

        for (std::size_t i = 0; i < 2; i++){
            if (res[i].second != ref[i].second){
                std::cout << "Neighbors: манхэттен индекс не совпал\n";
                return 1;
            }
            if (!approx_equal(res[i].first, ref[i].first)){
                std::cout << "Neighbors: манхэттен расстояние не совпало\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 3: k = rows — все объекты, порядок должен быть детерминирован
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 5, Weighting::Uniform);
        knn.fit(m, target_span);

        auto res = knn.find_neighbors(query, 5);
        if (res.size() != 5){
            std::cout << "Neighbors: k=5 вернул " << res.size() << " элементов\n";
            return 1;
        }
        for (std::size_t i = 0; i < 5; i++){
            if (res[i].second != i){
                std::cout << "Neighbors: порядок при k=5 нарушен на " << i << "\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 4: k=0 → InvalidK
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 1, Weighting::Uniform);
        knn.fit(m, target_span);

        try {
            knn.find_neighbors(query, 0);
            std::cout << "Neighbors: k=0 не бросил InvalidK\n";
            return 1;
        }
        catch (const InvalidK&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 5: k > rows → InvalidK
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 1, Weighting::Uniform);
        knn.fit(m, target_span);

        try {
            knn.find_neighbors(query, 100);
            std::cout << "Neighbors: k>rows не бросил InvalidK\n";
            return 1;
        }
        catch (const InvalidK&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 6: find_neighbors до fit → NotFitted
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);

        try {
            knn.find_neighbors(query, 3);
            std::cout << "Neighbors: до fit не бросил NotFitted\n";
            return 1;
        }
        catch (const NotFitted&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 7: query.size() != cols → DimensionMismatch
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);
        knn.fit(m, target_span);

        std::vector<double> bad_query = {1.0, 2.0, 3.0};    // 3 != 2
        try {
            knn.find_neighbors(std::span<const double>(bad_query), 3);
            std::cout << "Neighbors: несовпадение размерностей не бросило DimensionMismatch\n";
            return 1;
        }
        catch (const DimensionMismatch&){ /* OK */ }
    }

    std::cout << "Neighbors: все тесты пройдены\n";
    return 0;
}