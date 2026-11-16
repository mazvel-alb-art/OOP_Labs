#ifndef RANDOMFORESTREGRESSOR_HPP
#define RANDOMFORESTREGRESSOR_HPP

#include "IRegressor.hpp"
#include "CartRegressor.hpp"
#include "MaxFeatures.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <span>

/**
 * @brief Случайный лес CART для регрессии.
 *
 * Каждое дерево обучается на собственной bootstrap-выборке с кратностями;
 * предсказание леса является средним предсказаний деревьев.
 */
class RandomForestRegressor : public IRegressor{
private:
    std::size_t n_estimators_;
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;
    MaxFeatures	max_features_;
    bool bootstrap_;
    std::optional<std::size_t> max_samples_;
    std::uint32_t random_state_;
    std::size_t	n_jobs_;
    std::size_t	num_features_ = 0;
    bool fitted_ = false;
    std::vector<std::unique_ptr<CartRegressor>>	trees_;

public:
    /**
     * @brief Создаёт регрессор случайного леса.
     * @param n_estimators Число деревьев (не менее 1).
     * @param max_depth Максимальная глубина дерева; 0 означает отсутствие ограничения.
     * @param min_samples_split Минимум разных объектов для разделения узла (не менее 2).
     * @param min_samples_leaf Минимум разных объектов в листе (не менее 1).
     * @param max_features Режим выбора признаков в каждом узле.
     * @param bootstrap Использовать ли bootstrap-выборку объектов.
     * @param max_samples Число извлечений с возвращением; по умолчанию равно числу строк.
     * @param random_state Seed для воспроизводимого обучения.
     * @param n_jobs Максимальное число параллельных задач (не менее 1).
     * @throw std::invalid_argument Если параметры некорректны.
     */
    RandomForestRegressor(
        std::size_t n_estimators = 100,
        std::size_t max_depth = 0,
        std::size_t min_samples_split = 2,
        std::size_t min_samples_leaf = 1,
        MaxFeatures max_features = {},
        bool bootstrap = true,
        std::optional<std::size_t> max_samples = std::nullopt,
        std::uint32_t random_state = 42,
        std::size_t n_jobs = 1
    );

    /**
     * @brief Обучает лес на матрице признаков и целевых значениях.
     * @param X Матрица признаков, объекты расположены по строкам.
     * @param y Целевой отклик для каждой строки X.
     * @throw EmptyTrainingSet Если обучающая выборка пуста.
     * @throw DimensionMismatch Если размеры X, y или max_samples не согласованы.
     */
    void fit(const Matrix& X, std::span<const double> y) override;

    /**
     * @brief Усредняет предсказания всех деревьев для каждой строки X.
     * @throw NotFitted Если лес ещё не обучен.
     * @throw FeatureCountMismatch Если число признаков не совпадает с обучающим.
     */
    std::vector<double> predict(const Matrix& X) const override;
};

#endif
