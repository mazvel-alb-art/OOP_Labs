#ifndef MAXFEATURES_HPP
#define MAXFEATURES_HPP
#include <cstddef>

/** @brief Способ выбора признаков при поиске разбиения в узле дерева. */
enum class MaxFeaturesMode {
    All,   ///< Рассматривать все признаки.
    Sqrt,  ///< Рассматривать floor(sqrt(числа признаков)), не менее одного.
    Count  ///< Рассматривать число признаков, заданное в MaxFeatures::count.
};

/**
 * @brief Настройки случайного выбора признаков CART.
 *
 * Для режимов All и Sqrt поле count игнорируется.
 */
struct MaxFeatures {
    MaxFeaturesMode mode = MaxFeaturesMode::All; ///< Режим выбора признаков.
    std::size_t count = 0; ///< Число признаков для режима MaxFeaturesMode::Count.
};

#endif
