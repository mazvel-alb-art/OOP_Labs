#ifndef CARTREGRESSOR_HPP
#define CARTREGRESSOR_HPP

#include "IRegressor.hpp"
#include "Matrix.hpp"
#include <memory>
#include <vector>

/**
 * @brief Узел дерева регрессии.
 *
 * Либо лист (is_leaf == true), либо внутренний узел с потомками.
 */
struct CartNode {
    bool is_leaf = true;

    /// Для листа: среднее значение отклика в листе.
    double value = 0.0;

    /// Для внутреннего узла: номер признака.
    std::size_t feature = 0;
    /// Для внутреннего узла: порог.
    double threshold = 0.0;
    /// Левый потомок.
    std::unique_ptr<CartNode> left;
    /// Правый потомок.
    std::unique_ptr<CartNode> right;
};

/**
 * @brief Дерево решений CART для регрессии.
 */
class CartRegressor : public IRegressor {
private:
    std::size_t max_depth_;
    std::size_t min_samples_leaf_;
    std::size_t min_samples_split_;
    std::size_t num_features_ = 0;
    bool fitted = false;
    std::unique_ptr<CartNode> root;

    /**
     * @brief Создать лист по объектам узла.
     * @param y Вектор откликов.
     * @param object_ids Индексы объектов листа.
     * @return Указатель на лист.
     */
    std::unique_ptr<CartNode> make_leaf(
        std::span<const double> y,
        const std::vector<std::size_t>& object_ids) const;

    /**
     * @brief Рекурсивно построить узел.
     * @param X Матрица признаков.
     * @param y Вектор откликов.
     * @param object_ids Индексы объектов узла.
     * @param depth Текущая глубина (корень = 0).
     * @return Указатель на построенный узел.
     */
    std::unique_ptr<CartNode> build_node(
        const Matrix& X,
        std::span<const double> y,
        const std::vector<std::size_t>& object_ids,
        std::size_t depth);

    /**
     * @brief Предсказание для одной строки.
     * @param row Срез строки матрицы признаков.
     * @return Предсказанное значение.
     */
    double predict_one(MatrixSlice<const double> row) const;

public:
    /**
     * @brief Конструктор.
     * @param max_depth Максимальная глубина (0 = только корень-лист).
     * @param min_samples_split Минимум объектов для попытки разделить узел (>= 2).
     * @param min_samples_leaf Минимум объектов в каждом потомке (>= 1).
     * @throw std::invalid_argument Если параметры некорректны.
     */
    CartRegressor(std::size_t max_depth,
                  std::size_t min_samples_split,
                  std::size_t min_samples_leaf);

    /**
     * @brief Обучает дерево.
     * @param X Матрица признаков.
     * @param y Вектор откликов.
     */
    void fit(const Matrix& X, std::span<const double> y) override;

    /**
     * @brief Предсказывает отклики для матрицы.
     * @param X Матрица признаков.
     * @return Вектор предсказаний.
     */
    std::vector<double> predict(const Matrix& X) const override;

    /** @brief Число узлов в дереве. */
    std::size_t node_count() const;
    /** @brief Число листьев в дереве. */
    std::size_t leaf_count() const;
    /** @brief Фактическая максимальная глубина. */
    std::size_t actual_depth() const;
};

#endif