#include "Dataset.hpp"
#include <iostream>
#include <variant>

int main(){
    Dataset dataset("data/crx.csv");

    // Ожидаемые типы столбцов для crx.csv.
    // true = числовой, false = категориальный.
    std::vector<bool> expected = {
        false,  // A1: b
        true,   // A2: 30.83
        true,   // A3: 0
        false,  // A4: u
        false,  // A5: g
        false,  // A6: w
        false,  // A7: v
        true,   // A8: 1.25
        false,  // A9: t
        false,  // A10: t
        true,   // A11: 01
        false,  // A12: f
        false,  // A13: g
        true,   // A14: 00202
        true,   // A15: 0
        false   // A16: +
    };

    std::size_t n = dataset.get_column_count();
    if (n != expected.size()){
        std::cout <<"Missing_value: Ожидалось "<< expected.size()<<" признаков, получено "<<n<<std::endl;
        return 1;
    }

    for (std::size_t i = 0; i < n; i++){
        const Column& col = dataset.get_columns(i);
        bool is_numeric = std::holds_alternative<NumericColumn>(col);

        if (is_numeric != expected[i]){
            std::cout <<"Missing_value: Признак "<<i<<": ожидался "<<(expected[i] ? "числовой" : "категориальный")<< ", получен "<< (is_numeric ? "числовой" : "категориальный")<< std::endl;
            return 1;
        }
    }

    std::cout << "Missing_value: Тест пройден: все типы признаков определены верно\n";
    return 0;
}