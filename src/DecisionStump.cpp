#include "DecisionStump.hpp"
#include "Exceptions.hpp"

#include <numeric>
#include <algorithm>
#include <vector>
#include <limits>
#include <map>

/**
 * @brief Конструктор: инициализация критерия и min_samples_leaf.
 * @throw std::invalid_argument Если min_samples_leaf < 1.
 */
DecisionStump::DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf){
    if (min_samples_leaf < 1){
        throw std::invalid_argument("Неверное минимальное количсетво объектов для каждой стороны");
    }
        criterion_ = criterion;
        min_samples_leaf_ = min_samples_leaf;
}

/**
 * @brief Обучение пня. Проверки + развилка на fit_variance / fit_gini.
 * @throw EmptyTrainingSet Если выборка пустая.
 * @throw DimensionMismatch Если rows != target.size().
 */
void DecisionStump::fit(const Matrix& features, std::span<const double> target){
    //проверки
    if (features.rows() == 0){
        throw EmptyTrainingSet("Пустая матрица");
    }
    if (features.rows() != target.size()){
        throw DimensionMismatch("Несоответствие объектов и откликов");
    }
    if (features.rows() < 2 * min_samples_leaf_){
        has_split_ = false;
        double mean = 0;
        for (std::size_t i = 0; i < target.size();i++){
            mean += target[i];
        }
        left_value_ = right_value_ = mean / target.size();
        fitted_ = true;
        return;
    }
    train_cols_ = features.cols();

    if (criterion_ == SplitCriterion::Variance){
        fit_variance(features, target);
    }
    else{
        fit_gini(features, target);
    }
    fitted_ = true;
}

/**
 * @brief Обучение для регрессии. Критерий: max(S_L²/m_L + S_R²/m_R).
 *
 * Инкрементальное обновление: S_L += y, m_L++ при переносе объекта
 * из правой части в левую. Обход отсортированных индексов — O(n log n).
 */
void DecisionStump::fit_variance(const Matrix& features, std::span<const double> target){           //сортировка массива значений признака
    std::size_t n = features.rows();

    double S_total = 0.0;
    for (std::size_t i = 0; i < target.size();i++){
        S_total += target[i];
    }

    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t best_j = 0;
    double best_t = 0.0;
    bool found = false;

    for (std::size_t j = 0; j < features.cols();j++){
        std::vector<std::size_t> order;
        for (std::size_t i = 0;i < n;i++){
            order.push_back(i);
        }
        std::ranges::sort(order, [&](std::size_t a, std::size_t b){ //order — индексы объектов. features[a][j] — значение признака j у объекта a. Компаратор — «a раньше b, если X[a][j] < X[b][j]»
            return features[a][j] < features[b][j];
        });
        std::size_t m_L = 0;
        double S_L = 0.0;
        std::size_t m_R = n;
        double S_R = S_total;
        for (std::size_t p = 0; p  < n - 1 ;p++){
            m_L++;
            S_L += target[order[p]];
            m_R--;
            S_R -= target[order[p]];

            if (features[order[p]][j] == features[order[p+1]][j]){
                continue;;
            }
            if ((m_L < min_samples_leaf_) || (m_R < min_samples_leaf_)){
                continue;
            }
            double score = (S_L * S_L)/static_cast<double>(m_L) + (S_R * S_R)/static_cast<double>(m_R);

            double t = std::midpoint(features[order[p]][j], features[order[p+1]][j]);
            if (score > best_score){
                best_score = score;
                best_j = j;
                best_t = t;
                found = true;
            }
        }
    }

    if (found){
        has_split_ = true;
        split_threshold_ = best_t;
        split_feature_ = best_j;

        double sum_L = 0.0;
        double sum_R = 0.0;
        std::size_t count_L = 0;
        std::size_t count_R = 0;
        for (std::size_t i = 0; i < n;i++){
            if (features[i][split_feature_] <= best_t){
                sum_L += target[i];
                count_L++;
            }
            else{
                sum_R += target[i];
                count_R++;
            }
        }

        left_value_ = sum_L / static_cast<double>(count_L);
        right_value_ = sum_R / static_cast<double>(count_R);
    }
    else{
        has_split_ = false;
        double mean = 0.0;
        for (std::size_t i = 0;i < n;i++){
            mean += target[i];
        }
        left_value_ = right_value_ = mean/static_cast<double>(n);
    }    
}

/**
 * @brief Обучение для классификации. Критерий: max(A_L/m_L + A_R/m_R),
 *        где A = Σ_c n_c².
 *
 * Метки классов перенумеровываются через std::map (детерминизм).
 * Инкрементальное обновление: A_L += 2a+1, A_R -= 2b-1 при переносе
 * объекта класса c из правой части в левую.
 */
void DecisionStump::fit_gini(const Matrix& features, std::span<const double> target){//для классификации
    std::size_t n = features.rows();

    std::map<double, std::size_t> class_map; //исходное значение → номер
    std::vector<double> class_value;        //номер → исходное значение

    for (std::size_t i = 0;i < n;i++){
        if (class_map.find(target[i]) == class_map.end()){
            class_map[target[i]] = class_map.size();
            class_value.push_back(target[i]);
        }
    }
    std::size_t k = class_map.size();
    std::vector<std::size_t> class_of(n);       //номер класса для каждого объекта
    for (std::size_t i = 0; i < n;i++){
        class_of[i] = class_map[target[i]];
    }
    std::vector<std::size_t> counts_total(k, 0);
    for (std::size_t i = 0; i < n;i++){
        counts_total[class_of[i]]++;
    }
    std::size_t A_total = 0;
    for (std::size_t i = 0; i < k;i++){
        A_total += (counts_total[i] * counts_total[i]);
    }

    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t best_j = 0;
    double best_t = 0.0;
    bool found = false;
    for (std::size_t j = 0; j < features.cols();j++){
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0;i < n;i++){
            order[i] = i;
        }
        std::ranges::sort(order, [&](std::size_t a, std::size_t b){
        return features[a][j] < features[b][j];
        });
        std::vector<std::size_t> counts_L(k, 0);         // нули
        std::vector<std::size_t> counts_R = counts_total; // копия
        std::size_t A_L = 0;
        std::size_t A_R = A_total;
        std::size_t m_L = 0;
        std::size_t m_R = n;
        for (std::size_t p = 0; p < n - 1;p++){
            std::size_t c = class_of[order[p]];
            A_L += 2 * counts_L[c] + 1;
            A_R -= 2 * counts_R[c] - 1;
            m_L++;
            m_R--;
            counts_L[c]++;
            counts_R[c]--;

            if (features[order[p]][j] == features[order[p+1]][j]){
            continue;
            }
            if (m_L < min_samples_leaf_ || m_R < min_samples_leaf_){
            continue;
            }

            double score = static_cast<double>(A_L)/static_cast<double>(m_L) + static_cast<double>(A_R)/static_cast<double>(m_R);
            double t = std::midpoint(features[order[p]][j], features[order[p+1]][j]);
            if (score > best_score){
                best_score = score;
                best_j = j;
                best_t = t;
                found = true;
            }
        }
    }
    if (found){
        has_split_ = true;
        split_threshold_ = best_t;
        split_feature_ = best_j;

        std::vector<std::size_t> count_L(k, 0);
        std::vector<std::size_t> count_R(k, 0);
        for (std::size_t i = 0; i < n;i++){
            if (features[i][split_feature_] <= split_threshold_){
                count_L[class_of[i]]++;
            }
            else{
                count_R[class_of[i]]++;
            }
        }
        std::size_t best_L = 0;
        for (std::size_t l = 1;l < k;l++){
            if (count_L[best_L] < count_L[l]){
                best_L = l;
            }
        }
        left_value_ = class_value[best_L];

        std::size_t best_R = 0;
        for (std::size_t l = 1;l < k ;l++){
            if (count_R[best_R] < count_R[l]){
                best_R = l;
            }
        }
        right_value_ = class_value[best_R];
    }
    else{
        has_split_ = false;
        std::size_t best_c = 0;
        for (std::size_t c = 1; c < k; c++){
            if (counts_total[c] > counts_total[best_c]) {
            best_c = c;
            }
        }
        left_value_ = right_value_ = class_value[best_c];
    }
}

/**
 * @brief Предсказание для объектов.
 *
 * Для каждого объекта: X[i][split_feature_] <= split_threshold_
 * ? left_value_ : right_value_. Если has_split_ == false — вернуть
 * left_value_ (вырожденный случай).
 *
 * @throw NotFitted Если fit не вызывался.
 * @throw FeatureCountMismatch Если число признаков не совпадает.
 */
std::vector<double> DecisionStump::predict(const Matrix& features)const{
    if (fitted_ == false){
        throw NotFitted("fit пропущен!");
    }
    if (features.cols() != train_cols_){
        throw FeatureCountMismatch("Несоответствие количества призннаков");
    }
    std::size_t n = features.rows();
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n;i++){
        if (!has_split_){
            result[i] = left_value_;
        }
        else if (features[i][split_feature_] <= split_threshold_){
            result[i] = left_value_;
        }
        else{
            result[i] = right_value_;
        }
    }   
    return result;
}