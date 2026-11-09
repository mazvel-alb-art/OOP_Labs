#ifndef IREGRESSOR_HPP
#define IREGRESSOR_HPP

#include "IModel.hpp"

/**
 * @brief Общий контракт регрессоров.
 *
 * Дополнительных методов сейчас не требует — просто наследник IModel.
 */
class IRegressor : public IModel {
public:
    virtual ~IRegressor() = default;
};

#endif