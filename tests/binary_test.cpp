#include "Dataset.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

bool matrix_equal(const std::vector<std::vector<double>>& original, const std::vector<std::vector<double>>& copy){
    if (original.size() != copy.size()){
        return false;
    }
    for (std::size_t i = 0; i < original.size();i++){
        if (original[i].size() != copy[i].size()){
            return false;
        }
        for (std::size_t j = 0; j < original[i].size();j++){
            if (original[i][j] != copy[i][j]){
                return false;
            }
        }
    }
    return true;
}

int main(){
    Dataset dataset("data/crx.csv");
    dataset.preprocess();
    const auto original_matrix = dataset.get_matrix();
    dataset.save_binary("data/binary.bin");
    dataset.load_binary("data/binary.bin");
    const auto copy_matrix = dataset.get_matrix();
    if (!matrix_equal(original_matrix, copy_matrix)){
        std::cout<<"Binary: Тест провален"<<std::endl;
        return 1;
    }
    else{
        std::cout<<"Binary: Тест успешен"<<std::endl;
        return 0;
    }
}
