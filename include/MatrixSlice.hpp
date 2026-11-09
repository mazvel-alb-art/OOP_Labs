#ifndef MATRIXSLICE_HPP
#define MATRIXSLICE_HPP

#include "Exceptions.hpp"
#include <cstddef>
#include <span>

/**
 * @brief Направление среза: строка или столбец.
 */
enum class SliceAxis { Row, Column };

/**
 * @brief Невладеющий срез матрицы (строка или столбец).
 *
 * Хранит указатель на первый элемент, количество элементов, шаг в элементах
 * и направление среза. Позволяет работать со строками и столбцами независимо
 * от порядка хранения матрицы (RowMajor/ColumnMajor).
 *
 * @tparam T Тип элемента (`double` или `const double`).
 */
template<class T>
class MatrixSlice {
public:
    /** @brief Конструктор по умолчанию: пустой срез. */
    MatrixSlice() noexcept = default;

    /**
     * @brief Основной конструктор.
     * @param data Указатель на первый элемент среза.
     * @param size Число элементов среза.
     * @param stride Шаг между соседними элементами (в элементах, не байтах).
     * @param axis Направление среза (Row или Column).
     * @throw std::invalid_argument Если `data == nullptr` при `size > 0`
     *        или `stride == 0`.
     *
     * Если `size == 0`, указатель сохраняется как `nullptr` независимо от
     * переданного значения.
     */
    MatrixSlice(T* data, std::size_t size, std::size_t stride, SliceAxis axis) {
        if (size == 0) {
            data_ = nullptr;
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("MatrixSlice: data == nullptr for non-empty slice");
        }
        if (stride == 0) {
            throw std::invalid_argument("MatrixSlice: stride must be positive");
        }
        data_ = data;
        size_ = size;
        stride_ = stride;
        axis_ = axis;
    }

    /** @brief Число элементов в срезе. */
    std::size_t size() const noexcept { return size_; }

    /** @brief Пуст ли срез. */
    bool empty() const noexcept { return size_ == 0; }

    /** @brief Шаг между соседними элементами среза (в элементах). */
    std::size_t stride() const noexcept { return stride_; }

    /** @brief Направление среза (строка или столбец). */
    SliceAxis axis() const noexcept { return axis_; }

    /**
     * @brief Непрерывен ли срез в памяти.
     * @return `true`, если `size <= 1` или `stride == 1`.
     */
    bool is_contiguous() const noexcept {
        return size_ <= 1 || stride_ == 1;
    }

    /** @brief Указатель на первый элемент среза. */
    T* data() const noexcept { return data_; }

    /**
     * @brief Доступ к элементу по индексу (без проверки границ).
     * @param index Номер элемента в срезе (0..size-1).
     * @return Ссылка на элемент.
     */
    T& operator()(std::size_t index) const noexcept {
        return data_[index * stride_];
    }

    /**
     * @brief Проверяемый доступ к элементу.
     * @param index Номер элемента в срезе.
     * @return Ссылка на элемент.
     * @throw std::out_of_range Если индекс вне диапазона.
     */
    T& at(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("MatrixSlice::at: index out of range");
        }
        return data_[index * stride_];
    }

    /**
     * @brief Получить `std::span` на элементы среза.
     * @return `std::span<T>` длины `size()`.
     * @throw NonContiguousSliceError Если срез не непрерывен (stride != 1
     *        и size > 1).
     */
    std::span<T> operator*() const {
        if (!is_contiguous()) {
            throw NonContiguousSliceError("MatrixSlice: slice is not contiguous");
        }
        return std::span<T>(data_, size_);
    }

    /**
     * @brief Синоним `operator*()`.
     * @throw NonContiguousSliceError Если срез не непрерывен.
     */
    std::span<T> as_span() const {
        return *this;
    }

    /**
     * @brief Преобразовать в читаемый срез (const).
     * @return `MatrixSlice<const double>` над теми же элементами.
     */
    MatrixSlice<const double> as_const() const noexcept {
        return MatrixSlice<const double>(data_, size_, stride_, axis_);
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t stride_ = 1;
    SliceAxis axis_ = SliceAxis::Row;
};

#endif