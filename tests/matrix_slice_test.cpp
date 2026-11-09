#include "Matrix.hpp"
#include "Exceptions.hpp"
#include <iostream>
#include <cmath>

static int fail(const char* msg) {
    std::cout << "matrix_slice_test: FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // RowMajor 2x3: [1 2 3; 4 5 6]
    Matrix m;
    m.set_size(2, 3);
    m.data() = {1, 2, 3, 4, 5, 6};

    auto row0 = m(0, std::nullopt);
    if (row0.size() != 3) return fail("row size");
    if (row0.stride() != 1) return fail("row stride");
    if (row0.axis() != SliceAxis::Row) return fail("row axis");
    if (!row0.is_contiguous()) return fail("row contiguous");
    if (row0(0) != 1 || row0(1) != 2 || row0(2) != 3) return fail("row values");

    auto col1 = m(std::nullopt, 1);
    if (col1.size() != 2) return fail("col size");
    if (col1.stride() != 3) return fail("col stride");
    if (col1.axis() != SliceAxis::Column) return fail("col axis");
    if (col1.is_contiguous()) return fail("col not contiguous");
    if (col1(0) != 2 || col1(1) != 5) return fail("col values");

    // span из неплотного среза — исключение
    try {
        (void)*col1;
        return fail("expected NonContiguousSliceError");
    } catch (const NonContiguousSliceError&) {}

    // span из плотного среза
    auto sp = *row0;
    if (sp.size() != 3 || sp[0] != 1) return fail("row span");

    // изменение через срез
    row0(1) = 99;
    if (m(0, 1) != 99) return fail("row mutation");

    // ColumnMajor 2x3: same matrix
    Matrix cm;
    cm.set_size(2, 3);
    cm.set_layout(MatrixLayout::ColumnMajor);
    cm.data() = {1, 4, 2, 5, 3, 6};

    auto row1 = cm(1, std::nullopt);
    if (row1.stride() != 2) return fail("cm row stride");
    if (row1.is_contiguous()) return fail("cm row contiguous");
    if (row1(0) != 4 || row1(1) != 5 || row1(2) != 6) return fail("cm row values");

    auto col1_cm = cm(std::nullopt, 1);
    if (col1_cm.stride() != 1) return fail("cm col stride");
    if (!col1_cm.is_contiguous()) return fail("cm col contiguous");
    if (col1_cm(0) != 2 || col1_cm(1) != 5) return fail("cm col values");

    std::cout << "matrix_slice_test: OK\n";
    return 0;
}