#include "Dataset.hpp"
#include <iostream>
#include <variant>

int main(){
    Dataset dataset("data/crx.csv");

    std::size_t n = dataset.get_column_count();
    if (n == 0){
        std::cout << "Missing_value: Датасет пустой или не загрузился" << std::endl;
        return 1;
    }

    // Признаки, в которых по crx.csv точно есть пропуски.
    // Индексы взяты из column_order.
    std::vector<int> expected_missing = {
        0,   // A1
        1,   // A2
        3,   // A4
        4,   // A5
        5,   // A6
        6,   // A7
        13,  // A14

    };

    std::size_t total = 0;

    for (std::size_t i = 0; i < expected_missing.size(); i++){
        int idx = expected_missing[i];
        std::size_t count = dataset.get_missing_count(idx);
        total += count;

        if (count == 0){
            std::cout <<"Missing_value: Признак "<< idx<<" не содержит пропусков, хотя должен"<< std::endl;
            return 1;
        }
    }

    if (total == 0){
        std::cout << "Missing_value: Всего пропусков 0, хотя должны быть. Ошибка" << std::endl;
        return 1;
    }

    std::cout <<"Missing_value: Тест пройден, найдено пропусков: "<<total<<" в "<<expected_missing.size()<<" признаках"<<std::endl;
    return 0;
}