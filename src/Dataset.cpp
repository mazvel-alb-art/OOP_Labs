#include <iostream>
#include <variant> 
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>


#include "../include/Dataset.hpp"
#include "../include/NumericColumn.hpp"
#include "../include/CategoricalColumn.hpp"

Dataset::Dataset(const std::string file_way){
    std::ifstream file(file_way);
    std::string line1;
    std::string key;
    std::string buffer;
    std::string A;

    if (!file.is_open())
    {
        std::cout << "FILE NOT OPEN\n";
        return;
    }

    std::cout << "FILE OPENED\n";

    std::getline(file, line1);
    std::stringstream line2(line1);

    std::string x;
    int count = 0;

    while (std::getline(line2, buffer, ',')){
        count++;
    }//количество столбцов

    line2.clear();
    file.seekg(0);

    {//проверка согласованности файла
        bool correct = true;
        int i = 1;
        int j = 0;

        while (std::getline(file, line1)){
            line2.clear();
            line2.str(line1);

            while (std::getline(line2, buffer, ',')){
                j++;
            }

            if (j != count) {
                correct = false;
            }

            if (correct == false){
                break;
            }
            else {
                i++;
                j = 0;
            }
        }

        if (correct == false){
            std::cout<<"Файл не согласован! Ошибка на строке "<<i<<std::endl;
            return;
        }
        
    }

    file.clear();
    file.seekg(0);
    line2.clear();

    int number = 1;

    while (number <= count){//цикл для того чтобы взаимодействовать со всеми столбцами
        bool numeric = true;

        std::getline(file, line1);
        line2.clear();
        line2.str(line1);

        for (int i = 1;i <= count;i++){
            std::getline(line2, buffer, ',');

            if (i == number){
                A = buffer;//запомнить ключ к column
            }
        }

        line2.clear();

        while (std::getline(file, line1)){//цикл проверки на принадлежность столбца
            line2.clear();
            line2.str(line1);

            for (int i = 1; i <= count;i++){
                std::getline(line2, buffer, ',');

                if (i == number){
                    if (buffer == "?"){
                        continue;
                    }

                    try {
                        double x = std::stod(buffer);
                    }
                    catch (const std::invalid_argument ) {
                        numeric = false;
                        break;
                    }
                }
            }

            if (numeric == false){
                break;
            }
        }

        file.clear();
        line2.clear();
        file.seekg(0);

        if (numeric == true){
            std::getline(file, line1);
            NumericColumn arr;

            while (std::getline(file, line1)){
                line2.clear();
                line2.str(line1);

                for (int i = 1; i <= count;i++){
                    std::getline(line2, buffer, ',');                

                    if (i == number){
                        if (buffer == "?") {
                            arr.values.push_back(std::nullopt);
                        }
                        else{
                            arr.values.push_back(std::stod(buffer));
                        }
                    }
                }
            }

            columns[A] = arr;
            line2.clear();
            file.clear();
            file.seekg(0);
        }
        else {
            std::getline(file, line1);
            CategoricalColumn array;

            while (std::getline(file, line1)){
                line2.clear();
                line2.str(line1);            

                for (int i = 1; i <= count;i++){
                    std::getline(line2, buffer, ',');
                    
                    if (i == number){
                        if (buffer == "?") {
                            array.values.push_back(std::nullopt);
                        }
                        else{
                            array.values.push_back(buffer);
                        }
                    }
                }
            }

            columns[A] = array;
            line2.clear();
            file.clear();
            file.seekg(0);        
        }

        number++;
    }
}

//print_summary()

void Dataset::print_summary(){
    std::size_t column_lengh; 
    std::size_t column_count = columns.size();
    NumericColumn* x;
    CategoricalColumn* y;

    for (auto& column : columns){
        x = std::get_if<NumericColumn>(&column.second);

        if (x == nullptr){
            y = std::get_if<CategoricalColumn>(&column.second);
            column_lengh = y->values.size();
        }
        else{
            column_lengh = x->values.size();
        }
    }

    int* mass = new int[column_count];
    {
        int null_count = 0;
        std::size_t jindex = 0;

        for(auto& column : columns){
            x = std::get_if<NumericColumn>(&column.second);

            if (x != nullptr){
                for (std::size_t i = 0; i < column_lengh;i++){
                    if (x->values[i].has_value()){
                        continue;
                    }
                    else{
                        null_count++;
                    }
                }
            }
            else {
                y = std::get_if<CategoricalColumn>(&column.second);

                for (std::size_t i = 0; i < column_lengh;i++){
                    if (y->values[i].has_value()){
                        continue;
                    }
                    else{
                        null_count++;
                    }
                }
            }

            mass[jindex] = null_count;
            jindex++;
            null_count = 0;
        }
    }

    std::cout<<"Объектов: "<<column_lengh<<std::endl;
    std::cout<<"Признаков: "<<column_count<<std::endl;
    std::cout<<"#   Признак       Тип             Пропуски"<< std::endl;

    int all_null_count = 0;

    for (std::size_t i = 0;i < column_count;i++){
        all_null_count += mass[i];
    }

    int index = 1;

    for (auto& column : columns){
        std::cout<<index<<"  "<<column.first<<"            ";

        x = std::get_if<NumericColumn>(&column.second);

        if (x != nullptr){
            std::cout<<"Числовой        ";
        }
        else{
            std::cout<<"Категориальный  ";
        }

        double procent = static_cast<double>(mass[index]) / column_lengh * 100;
        std::cout<<mass[index]<<" ("<<procent<<"%)"<<std::endl;
        

        index++; 
    }
    std::cout<<"Всего пропусков: "<<all_null_count<<" в "<<column_count<<"признаках"<<std::endl;
    delete[] mass;
}

