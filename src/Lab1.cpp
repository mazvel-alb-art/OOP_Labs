#include <iostream>
#include <filesystem>
#include <limits>
#include "Dataset.hpp"

int main(){
    Dataset dataset("data/crx.csv");
    dataset.print_summary();

    while (true){
        std::cout <<"Введите номер признака (-1 для выхода): "<<std::endl;
        int x;
        if (!(std::cin >> x)){
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Некорректный ввод, попробуйте ещё раз"<<std::endl;
            continue;
        }
        if (x == -1) break;
        if (x < 0){
            std::cout << "Номер должен быть >= 0"<<std::endl;
            continue;
        }
        dataset.statistics(x);
    }

    std::cout <<"Сохранить предобработанный датасет? (yes/no): "<<std::endl;
    std::string answer;
    std::cin >> answer;

    if (answer == "yes" || answer == "Yes"){
        dataset.preprocess();
        dataset.save_binary("data/binary.bin");

        auto csv_size = std::filesystem::file_size("data/crx.csv");
        auto bin_size = std::filesystem::file_size("data/binary.bin");
        std::cout << "CSV:    " << csv_size << " байт"<<std::endl;
        std::cout << "Binary: " << bin_size << " байт"<<std::endl;
    }
    else if (answer == "no" || answer == "No"){
        std::cout << "Сохранение отменено"<<std::endl;
    }
    else{
        std::cout << "Некорректный ответ"<<std::endl;
    }

    return 0;
}