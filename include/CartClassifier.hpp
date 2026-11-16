#ifndef CARTCLASSIFIER_HPP
#define CARTCLASSIFIER_HPP

#include "IClassifier.hpp"
#include "Matrix.hpp"
#include "MaxFeatures.hpp"
#include <random>
#include <memory>
#include <vector>
#include <span>
#include <cstdint>

/**
 * @brief Узел дерева классификации.
 *
 * Лист хранит вектор вероятностей (длины classes_.size()). Внутренний узел
 * хранит номер признака, порог и двух потомков.
 */
struct CartClassifierNode {
    bool is_leaf = true;

    /// Для листа: вероятности классов.
    std::vector<double> probabilities;

    /// Для внутреннего узла: номер признака.
    std::size_t feature = 0;
    /// Для внутреннего узла: порог.
    double threshold = 0.0;
    /// Левый потомок.
    std::unique_ptr<CartClassifierNode> left;
    /// Правый потомок.
    std::unique_ptr<CartClassifierNode> right;
};

/**
 * @brief Дерево решений CART для классификации.
 */
class CartClassifier : public IClassifier {
private:
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    std::size_t num_features_ = 0;
    bool fitted_ = false;
    MaxFeatures max_features_;
    std::uint32_t random_state_;

    std::unique_ptr<CartClassifierNode> root_;
    std::vector<double> classes_;

    /**
     * @brief Создать лист по объектам узла (вектор вероятностей).
     * @param y Метки классов.
     * @param object_ids Исходные индексы объектов узла.
     * @param counts Кратности исходных объектов.
     * @return Новый лист с нормированными вероятностями классов.
     */
    std::unique_ptr<CartClassifierNode> make_leaf(
        std::span<const double> y,
        const std::vector<std::size_t>& object_ids,
        std::span<const std::size_t> counts) const;

    /**
     * @brief Рекурсивно построить узел.
     * @param X Матрица признаков.
     * @param y Метки классов.
     * @param object_ids Исходные индексы объектов узла.
     * @param depth Текущая глубина, корень имеет глубину 0.
     * @param counts Кратности исходных объектов.
     * @param rng Генератор случайного порядка признаков.
     * @return Новый лист или внутренний узел дерева.
     */
    std::unique_ptr<CartClassifierNode> build_node(
        const Matrix& X,
        std::span<const double> y,
        const std::vector<std::size_t>& object_ids,
        std::size_t depth,
        std::span<const std::size_t> counts,
        std::mt19937& rng);

    /**
     * @brief Вероятности классов для одной строки.
     * @param row Срез строки признаков.
     * @return Вероятности классов в порядке classes_.
     */
    std::span<const double> predict_proba_one(MatrixSlice<const double> row) const;

public:
    /**
     * @brief Конструктор.
     * @param max_depth Максимальная глубина (0 = без ограничения).
     * @param min_samples_split Минимум объектов для попытки разделить узел (>= 2).
     * @param min_samples_leaf Минимум объектов в каждом потомке (>= 1).
     * @param max_features Режим выбора признаков в каждом узле.
     * @param random_state Seed генератора случайного порядка признаков.
     * @throw std::invalid_argument Если параметры некорректны.
     */
    CartClassifier(std::size_t max_depth,
                   std::size_t min_samples_split,
                   std::size_t min_samples_leaf,
                    MaxFeatures max_features = {},
                    std::uint32_t random_state = 42);

    /**
     * @brief Обучает дерево, считая кратность каждой строки равной 1.
     * @param X Матрица признаков.
     * @param y Метки классов.
     */
    void fit(const Matrix& X, std::span<const double> y) override;

    /**
     * @brief Обучает дерево с целочисленными кратностями bootstrap-выборки.
     * @param X Матрица признаков.
     * @param y Метки классов, по одной на исходную строку.
     * @param counts Кратность каждой строки; нулевые значения исключают строку.
     * @throw EmptyTrainingSet Если выборка пуста или все кратности нулевые.
     * @throw DimensionMismatch Если размеры X, y и counts не согласованы.
     */
    void fit_with_counts(const Matrix& X, std::span<const double> y,
                         std::span<const std::size_t> counts);

    /**
     * @brief Предсказывает метку класса для каждой строки X.
     * @throw NotFitted Если дерево ещё не обучено.
     * @throw FeatureCountMismatch Если число признаков не совпадает с обучающим.
     */
    std::vector<double> predict(const Matrix& X) const override;

    /**
     * @brief Предсказывает вероятности классов.
     * @return Матрица X.rows() × classes().size(); столбцы соответствуют classes().
     * @throw NotFitted Если дерево ещё не обучено.
     * @throw FeatureCountMismatch Если число признаков не совпадает с обучающим.
     */
    Matrix predict_proba(const Matrix& X) const override;

    /**
     * @brief Возвращает список классов, встреченных в обучении, по возрастанию.
     * @throw NotFitted Если дерево ещё не обучено.
     */
    std::span<const double> classes() const override;

    /** @brief Число узлов в дереве. */
    std::size_t node_count() const;
    /** @brief Число листьев в дереве. */
    std::size_t leaf_count() const;
    /** @brief Фактическая максимальная глубина. */
    std::size_t actual_depth() const;
};

#endif
