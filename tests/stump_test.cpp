#include "Dataset.hpp"
#include "Matrix.hpp"
#include "DecisionStump.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <span>
#include <limits>
#include <numeric>
#include <map>
#include <stdexcept>

// ---------- вспомогательные ----------
bool approx_equal(double a, double b, double eps = 1e-6){
    return std::fabs(a - b) < eps;
}

// Матрица из вектора строк
Matrix make_matrix(const std::vector<std::vector<double>>& rows){
    Matrix m;
    if (rows.empty()) return m;
    std::size_t cols = rows[0].size();
    m.set_size(rows.size(), cols);
    for (const auto& r : rows){
        for (double v : r){
            m.data().push_back(v);
        }
    }
    return m;
}

// ---------- НАИВНАЯ реализация (только для теста) ----------
struct NaiveResult {
    bool found;
    std::size_t j;
    double t;
    double score;
};

// Variance: score = S_L²/m_L + S_R²/m_R, максимизируем
NaiveResult naive_variance(const Matrix& X, std::span<const double> y, std::size_t min_leaf){
    NaiveResult res{false, 0, 0.0, -std::numeric_limits<double>::infinity()};
    std::size_t n = X.rows();
    
    for (std::size_t j = 0; j < X.cols(); j++){
        std::vector<double> values;
        for (std::size_t i = 0; i < n; i++) values.push_back(X[i][j]);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        for (std::size_t k = 0; k + 1 < values.size(); k++){
            double t = std::midpoint(values[k], values[k+1]);
            double S_L = 0.0, S_R = 0.0;
            std::size_t m_L = 0, m_R = 0;
            for (std::size_t i = 0; i < n; i++){
                if (X[i][j] <= t){ S_L += y[i]; m_L++; }
                else              { S_R += y[i]; m_R++; }
            }
            if (m_L < min_leaf || m_R < min_leaf) continue;
            
            double score = S_L*S_L/static_cast<double>(m_L)
                         + S_R*S_R/static_cast<double>(m_R);
            if (score > res.score){
                res.found = true;
                res.j = j;
                res.t = t;
                res.score = score;
            }
        }
    }
    return res;
}

// Gini: score = A_L/m_L + A_R/m_R, максимизируем
NaiveResult naive_gini(const Matrix& X, std::span<const double> y, std::size_t min_leaf){
    NaiveResult res{false, 0, 0.0, -std::numeric_limits<double>::infinity()};
    std::size_t n = X.rows();
    
    for (std::size_t j = 0; j < X.cols(); j++){
        std::vector<double> values;
        for (std::size_t i = 0; i < n; i++) values.push_back(X[i][j]);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        for (std::size_t k = 0; k + 1 < values.size(); k++){
            double t = std::midpoint(values[k], values[k+1]);
            
            // счётчики классов слева/справа
            std::map<double, std::size_t> cntL, cntR;
            std::size_t m_L = 0, m_R = 0;
            for (std::size_t i = 0; i < n; i++){
                if (X[i][j] <= t){ cntL[y[i]]++; m_L++; }
                else              { cntR[y[i]]++; m_R++; }
            }
            if (m_L < min_leaf || m_R < min_leaf) continue;
            
            std::size_t A_L = 0, A_R = 0;
            for (const auto& [_, c] : cntL) A_L += c * c;
            for (const auto& [_, c] : cntR) A_R += c * c;
            
            double score = static_cast<double>(A_L)/static_cast<double>(m_L)
                         + static_cast<double>(A_R)/static_cast<double>(m_R);
            if (score > res.score){
                res.found = true;
                res.j = j;
                res.t = t;
                res.score = score;
            }
        }
    }
    return res;
}

int main(){
    // ============================================================
    // ТЕСТ 1. Регрессия: линейная зависимость
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {2}, {3}, {4}});
        std::vector<double> y = {10, 20, 30, 40};
        
        DecisionStump stump(SplitCriterion::Variance, 1);
        stump.fit(X, y);
        
        if (!stump.has_split()){
            std::cout << "stump_test: Variance линейная — разбиение не найдено\n";
            return 1;
        }
        if (stump.split_feature() != 0){
            std::cout << "stump_test: Variance линейная — неверный признак\n";
            return 1;
        }
        if (!approx_equal(stump.split_threshold(), 2.5, 0.01)){
            std::cout << "stump_test: Variance линейная — неверный порог: "
                      << stump.split_threshold() << "\n";
            return 1;
        }
        if (!approx_equal(stump.left_value(), 15.0, 0.01)){
            std::cout << "stump_test: Variance линейная — left_value: "
                      << stump.left_value() << "\n";
            return 1;
        }
        if (!approx_equal(stump.right_value(), 35.0, 0.01)){
            std::cout << "stump_test: Variance линейная — right_value: "
                      << stump.right_value() << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 2. Классификация: разделимо по одному признаку
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {2}, {3}, {4}, {5}, {6}});
        std::vector<double> y = {0, 0, 0, 1, 1, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        
        if (!stump.has_split()){
            std::cout << "stump_test: Gini — разбиение не найдено\n";
            return 1;
        }
        if (stump.split_feature() != 0){
            std::cout << "stump_test: Gini — неверный признак\n";
            return 1;
        }
        if (!approx_equal(stump.split_threshold(), 3.5, 0.01)){
            std::cout << "stump_test: Gini — неверный порог: "
                      << stump.split_threshold() << "\n";
            return 1;
        }
        if (!approx_equal(stump.left_value(), 0.0, 0.01)){
            std::cout << "stump_test: Gini — left_value\n";
            return 1;
        }
        if (!approx_equal(stump.right_value(), 1.0, 0.01)){
            std::cout << "stump_test: Gini — right_value\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 3. Наивная vs быстрая — Variance
    // ============================================================
    {
        Matrix X = make_matrix({
            {0.1}, {0.2}, {0.3}, {0.4}, {0.5},
            {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
        });
        std::vector<double> y = {1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
        
        DecisionStump stump(SplitCriterion::Variance, 2);
        stump.fit(X, y);
        
        NaiveResult ref = naive_variance(X, y, 2);
        
        if (stump.has_split() != ref.found){
            std::cout << "stump_test: naive Variance — несовпадение found\n";
            return 1;
        }
        if (stump.has_split()){
            if (stump.split_feature() != ref.j){
                std::cout << "stump_test: naive Variance — признак\n";
                return 1;
            }
            if (!approx_equal(stump.split_threshold(), ref.t, 1e-6)){
                std::cout << "stump_test: naive Variance — порог\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 4. Наивная vs быстрая — Gini
    // ============================================================
    {
        Matrix X = make_matrix({
            {0.1}, {0.2}, {0.3}, {0.4}, {0.5},
            {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
        });
        std::vector<double> y = {0, 0, 1, 1, 0, 1, 1, 0, 1, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 2);
        stump.fit(X, y);
        
        NaiveResult ref = naive_gini(X, y, 2);
        
        if (stump.has_split() != ref.found){
            std::cout << "stump_test: naive Gini — несовпадение found\n";
            return 1;
        }
        if (stump.has_split()){
            if (stump.split_feature() != ref.j){
                std::cout << "stump_test: naive Gini — признак\n";
                return 1;
            }
            if (!approx_equal(stump.split_threshold(), ref.t, 1e-6)){
                std::cout << "stump_test: naive Gini — порог\n";
                return 1;
            }
        }
    }

    // ============================================================
    // ТЕСТ 5. Константа — разбиения нет
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {1}, {1}, {1}});
        std::vector<double> y = {0, 0, 1, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        
        if (stump.has_split()){
            std::cout << "stump_test: константа — разбиение найдено, хотя не должно\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 6. Дубликаты: порог не между равными
    // ============================================================
    {
        Matrix X = make_matrix({{0}, {0}, {1}, {1}});
        std::vector<double> y = {0, 0, 1, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        
        if (!stump.has_split()){
            std::cout << "stump_test: дубликаты — разбиение не найдено\n";
            return 1;
        }
        // Единственный валидный порог — 0.5
        if (!approx_equal(stump.split_threshold(), 0.5, 0.01)){
            std::cout << "stump_test: дубликаты — порог не 0.5: "
                      << stump.split_threshold() << "\n";
            return 1;
        }
    }

    // ============================================================
    // ТЕСТ 7. predict до fit → NotFitted
    // ============================================================
    {
        DecisionStump stump(SplitCriterion::Gini, 1);
        Matrix X = make_matrix({{1}});
        
        try {
            stump.predict(X);
            std::cout << "stump_test: predict до fit — не бросил\n";
            return 1;
        }
        catch (const NotFitted&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 8. Разное число признаков → FeatureCountMismatch
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {2}});
        std::vector<double> y = {0, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        
        Matrix bad = make_matrix({{1, 2}});  // 2 признака вместо 1
        
        try {
            stump.predict(bad);
            std::cout << "stump_test: разное число признаков — не бросил\n";
            return 1;
        }
        catch (const FeatureCountMismatch&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 9. min_samples_leaf < 1 → std::invalid_argument
    // ============================================================
    {
        try {
            DecisionStump stump(SplitCriterion::Gini, 0);
            std::cout << "stump_test: min_samples_leaf=0 — не бросил\n";
            return 1;
        }
        catch (const std::invalid_argument&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 10. fit на пустой → EmptyTrainingSet
    // ============================================================
    {
        Matrix X;
        X.set_size(0, 1);
        std::vector<double> y;
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        
        try {
            stump.fit(X, y);
            std::cout << "stump_test: fit пустой — не бросил\n";
            return 1;
        }
        catch (const EmptyTrainingSet&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 11. fit с разной длиной → DimensionMismatch
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {2}, {3}});
        std::vector<double> y = {0, 1};   // 2 вместо 3
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        
        try {
            stump.fit(X, y);
            std::cout << "stump_test: fit разной длины — не бросил\n";
            return 1;
        }
        catch (const DimensionMismatch&){ /* OK */ }
    }

    // ============================================================
    // ТЕСТ 12. Через IModel& — как в приложении
    // ============================================================
    {
        Matrix X = make_matrix({{1}, {2}, {3}, {4}, {5}, {6}});
        std::vector<double> y = {0, 0, 0, 1, 1, 1};
        
        DecisionStump stump(SplitCriterion::Gini, 1);
        IModel& model = stump;
        
        model.fit(X, std::span<const double>(y));
        auto preds = model.predict(X);
        
        for (std::size_t i = 0; i < y.size(); i++){
            if (preds[i] != y[i]){
                std::cout << "stump_test: IModel& — неверное предсказание на "
                          << i << "\n";
                return 1;
            }
        }
    }

    std::cout << "stump_test: все тесты пройдены\n";
    return 0;
}