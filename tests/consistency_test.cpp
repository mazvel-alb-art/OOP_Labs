#include "Dataset.hpp"
#include <iostream>
#include <fstream>

int main(){
    // Создаём временный CSV с разным числом полей в строках.
    const std::string bad_file = "data/_bad.csv";
    {
        std::ofstream out(bad_file);
        out << "A1,A2,A3\n";
        out << "1,2,3\n";
        out << "1,2\n";          //здесь только 2 поля вместо 3
        out << "1,2,3\n";
    }

    Dataset dataset(bad_file);

    if (dataset.is_valid()){
        std::cout << "Consistency: тест провален — плохой CSV принят как валидный\n";
        std::remove(bad_file.c_str());
        return 1;
    }

    std::remove(bad_file.c_str());
    std::cout << "Consistency: тест пройден — плохой CSV отвергнут\n";
    return 0;
}