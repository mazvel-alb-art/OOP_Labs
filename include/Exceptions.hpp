#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

// Несовпадение размерностей (метрика, признаки).
class DimensionMismatch : public std::invalid_argument
{
    public:
    explicit DimensionMismatch(const std::string& msg)      //explicit конструктор не должен неявно конвертировать. Только явный вызов DimensionMismatch("...").
        : std::invalid_argument(msg) {}
};

// Некорректное значение k (k < 1 или k > размера выборки).
class InvalidK : public std::invalid_argument
{
    public:
    explicit InvalidK(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// Обучение на пустой выборке.
class EmptyTrainingSet : public std::invalid_argument
{
    public:
    explicit EmptyTrainingSet(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// При predict число признаков не совпало с обучающим.
class FeatureCountMismatch : public std::invalid_argument
{
    public:
    explicit FeatureCountMismatch(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// predict вызван до fit.
class NotFitted : public std::runtime_error
{
    public:
    explicit NotFitted(const std::string& msg)
        : std::runtime_error(msg) {}
};

#endif