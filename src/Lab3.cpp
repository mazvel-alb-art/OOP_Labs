#include "Dataset.hpp"
#include "Matrix.hpp"
#include "CartRegressor.hpp"
#include "CartClassifier.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <vector>
#include <span>
#include <iomanip>
#include <cmath>
#include <algorithm>

static void print_regressor_info(const CartRegressor& model) {
    std::cout << "  nodes:  " << model.node_count() << "\n";
    std::cout << "  leaves: " << model.leaf_count() << "\n";
    std::cout << "  depth:  " << model.actual_depth() << "\n";
}

static void print_classifier_info(const CartClassifier& model) {
    std::cout << "  nodes:  " << model.node_count() << "\n";
    std::cout << "  leaves: " << model.leaf_count() << "\n";
    std::cout << "  depth:  " << model.actual_depth() << "\n";
    std::cout << "  classes:";
    for (double c : model.classes()) std::cout << " " << c;
    std::cout << "\n";
}

int main() {
    try {
        Dataset dataset("data/crx.csv");
        dataset.print_summary();

        dataset.preprocess();
        std::vector<double> y = dataset.get_target_encoded();
        Split split = dataset.train_test_split(0.8);

        std::cout << "\nTrain: " << split.train_X.rows() << " объектов\n";
        std::cout << "Test:  " << split.test_X.rows() << " объектов\n\n";

        // ---------- Регрессия ----------
        std::cout << "=== CART Regressor (Variance) ===\n";
        CartRegressor reg(5, 10, 2);
        reg.fit(split.train_X, std::span<const double>(split.train_y));
        print_regressor_info(reg);

        auto reg_preds = reg.predict(split.test_X);
        std::cout << "  Первые 10 предсказаний (test):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, reg_preds.size()); ++i) {
            std::cout << "    pred=" << reg_preds[i]
                      << "  true=" << split.test_y[i] << "\n";
        }

        // MAE для регрессора
        double mae = 0.0;
        for (std::size_t i = 0; i < reg_preds.size(); ++i) {
            mae += std::fabs(reg_preds[i] - split.test_y[i]);
        }
        mae /= static_cast<double>(reg_preds.size());
        std::cout << "  MAE: " << mae << "\n";

        // ---------- Классификация ----------
        std::cout << "\n=== CART Classifier (Gini) ===\n";
        CartClassifier clf(5, 10, 2);
        clf.fit(split.train_X, std::span<const double>(split.train_y));
        print_classifier_info(clf);

        auto clf_preds = clf.predict(split.test_X);
        Matrix proba = clf.predict_proba(split.test_X);

        std::cout << "  Первые 10 предсказаний (test):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(10, clf_preds.size()); ++i) {
            std::cout << "    pred=" << clf_preds[i]
                      << "  true=" << split.test_y[i]
                      << "  proba=[";
            for (std::size_t c = 0; c < proba.cols(); ++c) {
                if (c) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(2) << proba(i, c);
            }
            std::cout << "]\n";
        }

        // Точность для классификатора
        std::size_t correct = 0;
        for (std::size_t i = 0; i < clf_preds.size(); ++i) {
            if (clf_preds[i] == split.test_y[i]) ++correct;
        }
        double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(clf_preds.size());
        std::cout << "  Точность: " << correct << " / " << clf_preds.size()
                  << " (" << acc << "%)\n";

        // ---------- Влияние глубины ----------
        std::cout << "\n=== Влияние max_depth (классификатор) ===\n";
        for (std::size_t d : {std::size_t(0), std::size_t(1), std::size_t(3), std::size_t(10)}) {
            CartClassifier m(d, 10, 2);
            m.fit(split.train_X, std::span<const double>(split.train_y));
            auto p = m.predict(split.test_X);
            std::size_t c = 0;
            for (std::size_t i = 0; i < p.size(); ++i) {
                if (p[i] == split.test_y[i]) ++c;
            }
            double a = 100.0 * static_cast<double>(c) / static_cast<double>(p.size());
            std::cout << "  max_depth=" << d
                      << "  nodes=" << m.node_count()
                      << "  leaves=" << m.leaf_count()
                      << "  actual_depth=" << m.actual_depth()
                      << "  accuracy=" << a << "%\n";
        }

    } catch (const std::exception& e) {
        std::cout << "Ошибка: " << e.what() << "\n";
        return 1;
    }
    return 0;
}