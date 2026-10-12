#include "KnnRegressor.hpp"
#include "KnnClassifier.hpp"
#include "EuclideanDistance.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <span>

// ------------------------------------------------------------
// Вспомогательные функции
// ------------------------------------------------------------

bool approx_equal(double a, double b, double eps = 1e-9){
    return std::fabs(a - b) < eps;
}

Matrix make_test_matrix(){
    Matrix m;
    m.set_size(5, 2);
    m.data() = {0, 0,
                1, 1,
                2, 2,
                3, 3,
                4, 4};
    return m;
}

std::vector<double> make_target_regression(){
    return {10, 20, 30, 40, 50};
}

std::vector<double> make_target_classification(){
    return {0, 0, 1, 1, 1};
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------

int main(){

    // ============================================================
    // ТЕСТ 1. k=1, регрессор, Uniform — на train возвращает те же значения
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_regression();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 1, Weighting::Uniform);
        knn.fit(X, std::span<const double>(target));

        auto preds = knn.predict(X);

        for (std::size_t i = 0; i < preds.size(); i++){
            if (!approx_equal(preds[i], target[i])){
                std::cout << "knn_test: k=1 регрессор не совпал на " << i << "\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 2. k=3, регрессор, Uniform — запрос [0,0] → среднее (10+20+30)/3 = 20
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_regression();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);
        knn.fit(X, std::span<const double>(target));

        Matrix query;
        query.set_size(1, 2);
        query.data() = {0, 0};

        auto preds = knn.predict(query);

        if (!approx_equal(preds[0], 20.0)){
            std::cout << "knn_test: k=3 Uniform ожидали 20, получили " << preds[0] << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 3. k=3, регрессор, Distance — exact match (запрос совпал с X[0])
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_regression();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Distance);
        knn.fit(X, std::span<const double>(target));

        Matrix query;
        query.set_size(1, 2);
        query.data() = {0, 0};   // совпадает с X[0] → d = 0

        auto preds = knn.predict(query);

        if (!approx_equal(preds[0], 10.0)){
            std::cout << "knn_test: Distance exact match ожидали 10, получили " << preds[0] << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 4. k=3, регрессор, Distance — обычный случай (не exact match)
    //   Запрос [0.5, 0.5]:
    //     к X[0] = sqrt(0.5)       ≈ 0.7071  → w ≈ 1.4142
    //     к X[1] = sqrt(0.5)       ≈ 0.7071  → w ≈ 1.4142
    //     к X[2] = sqrt(4.5)       ≈ 2.1213  → w ≈ 0.4714
    //   Взвешенное среднее = (1.4142*10 + 1.4142*20 + 0.4714*30) / 3.2998
    //                      ≈ 17.1429
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_regression();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Distance);
        knn.fit(X, std::span<const double>(target));

        Matrix query;
        query.set_size(1, 2);
        query.data() = {0.5, 0.5};

        auto preds = knn.predict(query);

        if (!approx_equal(preds[0], 17.1429, 0.001)){
            std::cout << "knn_test: Distance approx ожидали ~17.143, получили " << preds[0] << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 5. k=1, классификатор, Uniform — на train возвращает те же классы
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_classification();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnClassifier knn(std::move(metric), 1, Weighting::Uniform);
        knn.fit(X, std::span<const double>(target));

        auto preds = knn.predict(X);

        for (std::size_t i = 0; i < preds.size(); i++){
            if (preds[i] != target[i]){
                std::cout << "knn_test: k=1 классификатор не совпал на " << i << "\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 6. k=3, классификатор, Uniform
    //   target = {0, 0, 1, 1, 1}
    //   Запрос [0,0] → соседи X[0](0), X[1](0), X[2](1)
    //   Голоса: 0 → 2, 1 → 1 → победитель 0
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_classification();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnClassifier knn(std::move(metric), 3, Weighting::Uniform);
        knn.fit(X, std::span<const double>(target));

        Matrix query;
        query.set_size(1, 2);
        query.data() = {0, 0};

        auto preds = knn.predict(query);

        if (preds[0] != 0.0){
            std::cout << "knn_test: k=3 классификатор ожидали 0, получили " << preds[0] << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 7. k=3, классификатор, Distance — exact match
    //   Запрос [1,1] совпадает с X[1] (класс 0)
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_classification();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnClassifier knn(std::move(metric), 3, Weighting::Distance);
        knn.fit(X, std::span<const double>(target));

        Matrix query;
        query.set_size(1, 2);
        query.data() = {1, 1};

        auto preds = knn.predict(query);

        if (preds[0] != 0.0){
            std::cout << "knn_test: Distance exact match класс ожидали 0, получили " << preds[0] << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 8. predict до fit → NotFitted
    // ============================================================
    {
        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);

        Matrix X = make_test_matrix();

        try {
            knn.predict(X);
            std::cout << "knn_test: predict до fit не бросил\n";
            return 1;
        }
        catch (const NotFitted&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 9. Разное число признаков → FeatureCountMismatch
    // ============================================================
    {
        Matrix X = make_test_matrix();
        auto target = make_target_regression();

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);
        knn.fit(X, std::span<const double>(target));

        Matrix bad_query;
        bad_query.set_size(1, 5);
        bad_query.data() = {1, 2, 3, 4, 5};   // 5 != 2

        try {
            knn.predict(bad_query);
            std::cout << "knn_test: несовпадение признаков не бросило\n";
            return 1;
        }
        catch (const FeatureCountMismatch&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 10. fit на пустой выборке → EmptyTrainingSet
    // ============================================================
    {
        Matrix empty_X;
        empty_X.set_size(0, 2);
        std::vector<double> empty_target;

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);

        try {
            knn.fit(empty_X, std::span<const double>(empty_target));
            std::cout << "knn_test: пустая выборка не бросила\n";
            return 1;
        }
        catch (const EmptyTrainingSet&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 11. k=0 в конструкторе → InvalidK
    // ============================================================
    {
        try {
            auto metric = std::make_unique<EuclideanDistance>();
            KnnRegressor knn(std::move(metric), 0, Weighting::Uniform);
            std::cout << "knn_test: k=0 не бросил\n";
            return 1;
        }
        catch (const InvalidK&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 12. fit: features.rows() != target.size() → DimensionMismatch
    // ============================================================
    {
        Matrix X = make_test_matrix();          // 5 строк
        std::vector<double> bad_target = {1, 2, 3};   // 3 значения

        auto metric = std::make_unique<EuclideanDistance>();
        KnnRegressor knn(std::move(metric), 3, Weighting::Uniform);

        try {
            knn.fit(X, std::span<const double>(bad_target));
            std::cout << "knn_test: rows != target.size не бросило\n";
            return 1;
        }
        catch (const DimensionMismatch&){ /* OK */ }
    }

    std::cout << "knn_test: все тесты пройдены\n";
    return 0;
}