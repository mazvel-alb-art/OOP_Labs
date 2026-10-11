#include "Dataset.hpp"
#include "Matrix.hpp"

#include <iostream>
#include <vector>
#include <string>

bool matrix_equal(const Matrix& a, const Matrix& b){
    if (a.rows() != b.rows()) return false;
    if (a.cols() != b.cols()) return false;

    const auto& da = a.data();
    const auto& db = b.data();
    if (da.size() != db.size()) return false;

    for (std::size_t i = 0; i < da.size(); i++){
        if (da[i] != db[i]) return false;
    }
    return true;
}

int main(){
    Dataset dataset("data/crx.csv");
    Matrix& original = dataset.preprocess();

    original.save_binary("data/binary.bin");

    Matrix copy;
    copy.load_binary("data/binary.bin");

    if (!matrix_equal(original, copy)){
        std::cout << "Binary: Тест провален" << std::endl;
        return 1;
    }
    else{
        std::cout << "Binary: Тест успешен" << std::endl;
        return 0;
    }
}