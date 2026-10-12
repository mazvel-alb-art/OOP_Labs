#include "Dataset.hpp"
#include "Matrix.hpp"
#include "EuclideanDistance.hpp"
#include "ManhattanDistance.hpp"
#include "ChebyshevDistance.hpp"
#include "KnnClassifier.hpp"
#include "KnnRegressor.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <span>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Эксперимент с классификатором: обучаем, предсказываем, считаем точность
// ---------------------------------------------------------------------------
void run_classifier(
    const Split& split,
    std::unique_ptr<IDistance> metric,
    std::size_t k,
    Weighting weighting,
    const std::string& label)
{
    KnnClassifier knn(std::move(metric), k, weighting);
    knn.fit(split.train_X, std::span<const double>(split.train_y));
    auto preds = knn.predict(split.test_X);

    std::size_t correct = 0;
    for (std::size_t i = 0; i < preds.size(); i++){
        if (preds[i] == split.test_y[i]){
            correct++;
        }
    }
    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(preds.size());

    std::cout << "=== " << label << " ===\n";
    std::cout << "  Точность: " << correct << " / " << preds.size()
              << " (" << accuracy << "%)\n";
    std::cout << "  Первые 5 предсказаний:\n";
    for (std::size_t i = 0; i < 5; i++){
        std::cout << "    pred = " << preds[i]
                  << "   actual = " << split.test_y[i] << "\n";
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Эксперимент с регрессором: предсказывает числа 0..1 (не только 0 или 1)
// ---------------------------------------------------------------------------
void run_regressor(
    const Split& split,
    std::unique_ptr<IDistance> metric,
    std::size_t k,
    Weighting weighting,
    const std::string& label)
{
    KnnRegressor reg(std::move(metric), k, weighting);
    reg.fit(split.train_X, std::span<const double>(split.train_y));
    auto preds = reg.predict(split.test_X);

    std::cout << "=== " << label << " ===\n";
    std::cout << "  Первые 10 предсказаний (числа):\n";
    for (std::size_t i = 0; i < 10; i++){
        std::cout << "    pred = " << preds[i]
                  << "   actual = " << split.test_y[i] << "\n";
    }
    std::cout << std::endl;
}

int main(){
    try {
        Dataset dataset("data/crx.csv");
        dataset.print_summary();

        Matrix& X = dataset.preprocess();
        std::vector<double> y = dataset.get_target_encoded();

        std::cout << "Матрица признаков: " << X.rows() << " x " << X.cols() << "\n";
        std::cout << "Отклик: " << y.size() << " значений\n\n";

        Split split = dataset.train_test_split(0.8);
        std::cout << "Train: " << split.train_X.rows() << " объектов\n";
        std::cout << "Test:  " << split.test_X.rows() << " объектов\n\n";

        // -------- Классификатор: разные k, метрики, взвешивание --------
        run_classifier(split,
                       std::make_unique<EuclideanDistance>(),
                       3, Weighting::Uniform,
                       "Классификатор: k=3, Euclidean, Uniform");

        run_classifier(split,
                       std::make_unique<EuclideanDistance>(),
                       3, Weighting::Distance,
                       "Классификатор: k=3, Euclidean, Distance");

        run_classifier(split,
                       std::make_unique<ManhattanDistance>(),
                       5, Weighting::Uniform,
                       "Классификатор: k=5, Manhattan, Uniform");

        run_classifier(split,
                       std::make_unique<ChebyshevDistance>(),
                       7, Weighting::Distance,
                       "Классификатор: k=7, Chebyshev, Distance");

        run_classifier(split,
                       std::make_unique<EuclideanDistance>(),
                       15, Weighting::Uniform,
                       "Классификатор: k=15, Euclidean, Uniform");

        // -------- Регрессор: видно, как меняются числа при разных k --------
        run_regressor(split,
                      std::make_unique<EuclideanDistance>(),
                      1, Weighting::Uniform,
                      "Регрессор: k=1, Euclidean, Uniform");

        run_regressor(split,
                      std::make_unique<EuclideanDistance>(),
                      3, Weighting::Uniform,
                      "Регрессор: k=3, Euclidean, Uniform");

        run_regressor(split,
                      std::make_unique<EuclideanDistance>(),
                      15, Weighting::Uniform,
                      "Регрессор: k=15, Euclidean, Uniform");

        run_regressor(split,
                      std::make_unique<EuclideanDistance>(),
                      3, Weighting::Distance,
                      "Регрессор: k=3, Euclidean, Distance");
    }
    catch (const std::exception& e){
        std::cout << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}