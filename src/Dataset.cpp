#include <iostream>
#include <variant> 
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>


#include "Dataset.hpp"
#include "NumericColumn.hpp"
#include "CategoricalColumn.hpp"
#include "Statistics.hpp"

Dataset::Dataset(const std::string file_way){
    
    std::ifstream file(file_way);
    if (!file.is_open())
    {
        std::cout << "FILE NOT OPEN\n";
        return;
    }
    std::cout << "FILE OPENED\n";
    std::string line1;
    std::string key;
    std::string buffer;
    std::string A;


    std::getline(file, line1);
    std::stringstream line2(line1);

    std::string x;
    int count = 0;

    while (std::getline(line2, buffer, ',')){
        count++;
    }//количество столбцов
    statistic.column_count = count;

    line2.clear();
    file.seekg(0);

    if (!(Dataset::consistency_check(file, line1, buffer, count))){     //проверка согласованности + заполнение statistic.column_lenght
        return;
    }


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
                column_order.push_back(A);
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
                        (void)std::stod(buffer);
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

    for (auto& column : columns){       //ищем количество колонн и их длину
        x = std::get_if<NumericColumn>(&column.second);

        if (x == nullptr){
            y = std::get_if<CategoricalColumn>(&column.second);
            column_lengh = y->values.size();
        }
        else{
            column_lengh = x->values.size();
        }
    }

    std::vector<int> mass(column_count);
    {
        int null_count = 0;

        for(std::size_t jindex = 0; jindex < column_count;jindex++){        //ищем количество пропусков для каждого признака
            auto& column = columns[column_order[jindex]];
            x = std::get_if<NumericColumn>(&column);

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
                y = std::get_if<CategoricalColumn>(&column);

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

    for (std::size_t i = 0;i < column_count;i++){       //подсчёт всех пропусков
        all_null_count += mass[i];
    }

    int index = 0;

    for (auto& key : column_order){
        auto& column = columns[key];
        std::cout<<index<<"  "<<key<<"            ";

        x = std::get_if<NumericColumn>(&column);

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
    std::cout<<"Всего пропусков: "<<all_null_count<<" в "<<column_count<<" признаках"<<std::endl;
}







//statistic()

void Dataset::statistics(int x){
    std::size_t column_count = columns.size();

    NumericColumn* nume;
    CategoricalColumn* cate;

    int null_count = 0;
    double min;
    double max;
    double mean = 0;
    double var = 0;
    double q05 = 0;
    double q25 = 0;
    double q50 = 0;
    double q75 = 0;
    double q95 = 0;

    int interval1 = 0;
    int interval2 = 0;
    int interval3 = 0;
    int interval4 = 0;





    if (x < 0 || static_cast<std::size_t>(x) >= column_count){
        return;
    }
    std::string key = column_order[x];
    auto& column = columns[key];
    nume = std::get_if<NumericColumn>(&column);
    if (nume != nullptr){       //если числововй

        null_count = Dataset::null_count(nume);                     //количество пропусков

        Dataset::min_max_mean(nume, min, max, mean, null_count);   //минимум, макимум и медиана

        Dataset::var(nume, var, mean, null_count);                 //дисперсия

        Dataset::q(nume, q05, q25, q50, q75, q95);                 //квантили

        Dataset::intervals(nume, interval1, interval2, interval3, interval4); //интервалы


        

        std::cout<<"Признак "<<x<<": "<<key<<" числовой"<<std::endl;

        std::cout<<"Пропусков: "<<null_count<<std::endl;

        std::cout<<"min:   "<<min<<"   max:    "<<max<<std::endl;

        std::cout<<"mean:  "<<mean<<"   var:    "<<var<<std::endl<<std::endl;

        std::cout<<"q05:  "<<q05<<std::endl;
        std::cout<<"q25:  "<<q25<<std::endl;
        std::cout<<"q50:  "<<q50<<std::endl;
        std::cout<<"q75:  "<<q75<<std::endl;
        std::cout<<"q95:  "<<q95<<std::endl<<std::endl;

        std::cout<<"["<<0<<", "<<"25.0"<<")    ";
        for (int i = 0; i < interval1 / 10;i++){
            std::cout<<"#";
        }
        std::cout<<"    "<<interval1<<std::endl;

        std::cout<<"["<<"25.0"<<", "<<"50.0"<<")    ";
        for (int i = 0; i < interval2 / 10;i++){
            std::cout<<"#";
        }
        std::cout<<"    "<<interval2<<std::endl;

        std::cout<<"["<<"50.0"<<", "<<"80.0"<<")    ";
        for (int i = 0; i < interval3 / 10;i++){
            std::cout<<"#";
        }
        std::cout<<"    "<<interval3<<std::endl;

        std::cout<<"["<<"80.0"<<", "<<"+∞"<<")    ";
        for (int i = 0; i < interval4 / 10;i++){
            std::cout<<"#";
        }
        std::cout<<"    "<<interval4<<std::endl;

    }









    else{       //если категориальный
        cate = std::get_if<CategoricalColumn>(&column);         //Пропуски

        null_count = Dataset::null_count(cate);


        std::vector<std::string> category_list;         
        std::vector<int> periodicity;
        Dataset::category_list_periodicity(cate, category_list, periodicity);    //Заполнение списка уникальных категорий и частот категорий


        std::cout<<"Признак "<<x<<": "<<key<<" категориальный"<<std::endl;

        std::cout<<"Пропусков: "<<null_count<<std::endl;

        for (std::size_t i = 0;i < category_list.size();i++){
            std::cout<<category_list[i]<<"    ";
            for (int j = 0;j < (periodicity[i])/10;j++){
                 std::cout<<"#";
            }
            std::cout<<"   "<<periodicity[i]<<std::endl;
        }
                
    }
}

//

bool Dataset::consistency_check(std::ifstream& file, std::string line1, std::string buffer, int count){
    std::stringstream line3(line1);
    bool correct = true;
    int max_i = 1;
    int i = 1;
    int j = 0;

    while (std::getline(file, line1)){
        line3.clear();
        line3.str(line1);

        while (std::getline(line3, buffer, ',')){
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
        if (i > max_i){
            max_i = i;
        }
    }

    
    file.clear();
    file.seekg(0);
    line3.clear();
    if (correct == false){
        std::cout<<"Файл не согласован! Ошибка на строке "<<i<<std::endl;
        return correct;
    }
    else{
        statistic.column_lenght = max_i - 2;
        return correct;
    }    
}

// 

std::size_t Dataset::null_count(NumericColumn* nume){
    std::size_t null_count = 0;
    for (std::size_t i = 0;i < statistic.column_lenght;i++){
        if (nume->values[i].has_value()){
            continue;
        }
        else{
            null_count++;
        }
    }
    return null_count;
}

//

void Dataset::min_max_mean(NumericColumn* nume, double& min, double& max, double& mean, int& null_count){
bool start = true;
    for (std::size_t i = 0; i < statistic.column_lenght;i++){

        if (start == true){
            if (nume->values[i].has_value()){
                min = nume->values[i].value();
                max = nume->values[i].value();
                start = false;
            }
            else{
                continue;
            }
        }
        if (nume->values[i].has_value()){
            if (nume->values[i].value() > max){
                max = nume->values[i].value();
            }
        }
        

        if (nume->values[i].has_value()){
            if (nume->values[i].value() < min){
                min = nume->values[i].value();
            }
        }
        

        if (nume->values[i].has_value()){
            mean += nume->values[i].value();
            }
    }
    mean = mean/(statistic.column_lenght - null_count);   
}

//

void Dataset::var(NumericColumn* nume, double& var, double& mean, int& null_count){
    for (std::size_t i = 0; i < statistic.column_lenght; i++){
        if (nume->values[i].has_value()){
            var += (nume->values[i].value() - mean) * (nume->values[i].value() - mean);
        }
    }
    var = var / (statistic.column_lenght - null_count);
}

//

void Dataset::q(NumericColumn* nume, double& q05, double& q25, double& q50, double& q75, double& q95){
    std::vector<double> sort;
    for (std::size_t i = 0;i < statistic.column_lenght ;i++){       //сортировка
        if(nume->values[i].has_value()){
            sort.push_back(nume->values[i].value());
        }
    }
    for (std::size_t i = 0; i < sort.size()-1;i++){
        for (std::size_t j = 0; j < sort.size()-1;j++){
            if (sort[j] > sort[j+1]){
                double buffer = sort[j];
                sort[j] = sort[j+1];
                sort[j+1] = buffer;
            }
        }
    }

    q05 = sort[static_cast<int>(sort.size()*0.05 - 1)];       //квантили
    q25 = sort[static_cast<int>(sort.size()*0.25 - 1)];

    if (sort.size() % 2 == 0){
        q50 = (sort[static_cast<int>(sort.size()/2 - 1)] + sort[static_cast<int>(sort.size()/2)])/2;
    }
    else{
        q50 = sort[static_cast<int>(sort.size()/2)];
    }

    q75 = sort[static_cast<int>(sort.size()*0.75 - 1)];
    q95 = sort[static_cast<int>(sort.size()*0.95 - 1)];

}

//

void Dataset::intervals(NumericColumn* nume, int& interval1, int& interval2, int& interval3, int& interval4){
    for (std::size_t i = 0; i < statistic.column_lenght;i++){
        if (nume->values[i].has_value()){
            if (nume->values[i].value() >= 0 && nume->values[i].value() < 25){
                interval1++;
            }
            if (nume->values[i].value() >= 25 && nume->values[i].value() < 50){
                interval2++;
            }
            if (nume->values[i].value() >= 50 && nume->values[i].value() < 80){
                interval3++;
            }
            if (nume->values[i].value() >= 80 ){
                interval4++;
            }
        }
    }
}

//

std::size_t Dataset::null_count(CategoricalColumn* cate){
    std::size_t null_count = 0;
    for (std::size_t i = 0; i < statistic.column_lenght;i++){
        if (cate->values[i].has_value()){
            continue;
        }
        else{
            null_count++;
        }
    }
    return null_count;
}

//

void Dataset::category_list_periodicity(CategoricalColumn* cate, std::vector<std::string>& category_list, std::vector<int>& periodicity){
    bool in = false;
    for (std::size_t i = 0; i < statistic.column_lenght;i++){      //заполнение списка уникальных категорий
        for (std::size_t j = 0; j < category_list.size();j++){
            if (!cate->values[i].has_value()){
            continue;
            }
            else if (cate->values[i] == category_list[j]){
                in = true;
            }
        }
        if (!cate->values[i].has_value()){
            continue;
        }
        else if (in == false){
            category_list.push_back(cate->values[i].value());
        }
        in = false;
    }

    periodicity.resize(category_list.size());
    int periodicity_count = 0;
    for (std::size_t i = 0;i < category_list.size();i++){                   //ищем частоты категорий
        for (std::size_t j = 0;j < statistic.column_lenght;j++){
            if (!cate->values[j].has_value()){
                continue;
            }
            if (category_list[i] == cate->values[j])
            periodicity_count++;
        }
        periodicity[i] = periodicity_count;
        periodicity_count = 0;
    }
}

//

void Dataset::q50(){
    statistic.q50.clear();
    NumericColumn* nume;
    double q50;
    for (std::size_t k = 0; k < statistic.column_count;k++){  
    std::string key = column_order[k];
    nume = std::get_if<NumericColumn>(&columns[key]);
    if (nume != nullptr){      
    std::vector<double> sort;
    for (std::size_t i = 0;i < statistic.column_lenght ;i++){       //сортировка
        if(nume->values[i].has_value()){
            sort.push_back(nume->values[i].value());
        }
    }
    for (std::size_t i = 0; i < sort.size()-1;i++){
        for (std::size_t j = 0; j < sort.size()-1;j++){
            if (sort[j] > sort[j+1]){
                double buffer = sort[j];
                sort[j] = sort[j+1];
                sort[j+1] = buffer;
            }
        }
    }


    if (sort.size() % 2 == 0){
        q50 = (sort[static_cast<int>(sort.size()/2 - 1)] + sort[static_cast<int>(sort.size()/2)])/2;
    }
    else{
        q50 = sort[static_cast<int>(sort.size()/2)];
    }


    statistic.q50.push_back(q50);
}
    else{
        statistic.q50.push_back(0.0);
    }
}
}


void Dataset::preprocess(){
    CategoricalColumn* cate;
    NumericColumn* nume;
    Dataset::q50();
    std::vector<std::vector<std::string>> category_list(statistic.column_count);
    for (std::size_t i = 0; i < statistic.column_count;i++){        //подсчёт будующего количества колонн
        std::string key = column_order[i];
        auto& column = columns[key];
        nume = std::get_if<NumericColumn>(&column);
        if (nume != nullptr){
        }
        else {
            cate = std::get_if<CategoricalColumn>(&column);
            bool in = false;
            for (std::size_t j = 0; j < statistic.column_lenght;j++){
                if (cate->values[j].has_value()){
                    for (std::size_t  l = 0; l < category_list[i].size();l++){
                        if (cate->values[j].value() == category_list[i][l]){
                            in = true;
                        }
                    }
                    if (in == false){
                        category_list[i].push_back(cate->values[j].value());
            
                    }
                }
                else{
                    for (std::size_t l = 0; l < category_list[i].size();l++){
                        if (category_list[i][l] == "__MISSING__"){
                            in = true;
                        }

                    }
                    if (in == false){
                    category_list[i].push_back("__MISSING__");
                }
                in = false;
                }
                in = false;
            

            }
        }
    }



    matrix.resize(statistic.column_lenght);     
    std::vector<std::vector<std::variant<std::string, double>>> filled(statistic.column_lenght);

    for (std::size_t i = 0; i < statistic.column_lenght;i++){        //заполнение пропусков
            for (std::size_t j = 0;j < statistic.column_count;j++){
                std::string key = column_order[j];
                nume = std::get_if<NumericColumn>(&columns[key]);

                if (nume != nullptr){       //если колонна числовая
                    if (nume->values[i].has_value()){
                        filled[i].push_back(nume->values[i].value());
                    }
                    else{
                        filled[i].push_back(statistic.q50[j]);
                    }
                }

            
        

                else{                       //если колонна категориальная
                    cate = std::get_if<CategoricalColumn>(&columns[key]);
                        if (cate->values[i].has_value()){
                            filled[i].push_back(cate->values[i].value());
                        }
                        else{
                            filled[i].push_back("__MISSING__");
                        }
                }
            }
    }

    std::vector<double> mass_min(statistic.column_count);
    std::vector<double> mass_max(statistic.column_count);

    {                                   //минимумы и максимумы для каждой колонный
        for (std::size_t i = 0; i < statistic.column_count;i++){
            double min;
            double max;
            bool start = false;
            std::string key = column_order[i];
            nume = std::get_if<NumericColumn>(&columns[key]);
            if (nume != nullptr){
                for (std::size_t j = 0; j < statistic.column_lenght;j++){
                    if (nume->values[j].has_value()){
                        if (start == false){
                            min = nume->values[j].value();
                            max = nume->values[j].value();
                            start = true;
                        }
                        if (nume->values[j].value() > max){
                            max = nume->values[j].value();
                        }
                        if (nume->values[j].value() < min){
                            min = nume->values[j].value();
                        }
                    }
                }
            }
            else{
                continue;
            }
            mass_min[i] = min;
            mass_max[i] = max;
        }
    }

    
    for (std::size_t i = 0; i < statistic.column_count;i++){                               //масштабирование числовых призанков и кодирование категориальных
        std::string key = column_order[i];
        nume = std::get_if<NumericColumn>(&columns[key]);

        if (nume != nullptr){
            matrix_column_names.push_back(key);
            for (std::size_t j = 0; j < statistic.column_lenght;j++){
                double scaled = ((std::get<double>(filled[j][i])) - mass_min[i]) / (mass_max[i] - mass_min[i]);
                matrix[j].push_back(scaled);
            }
        }
        else{
            for (std::size_t k = 0; k < category_list[i].size();k++){
                matrix_column_names.push_back(key + "_" + category_list[i][k]);
                for (std::size_t j = 0; j < statistic.column_lenght; j++){
                    if (category_list[i][k] == (std::get<std::string>(filled[j][i]))){
                        matrix[j].push_back(1);
                    }
                    else{
                        matrix[j].push_back(0);
                    }
                }
            }
        }
    }
}



void Dataset::save_binary(const std::string& file_name){
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для записи: " << file_name <<std::endl;
        return;
    }
    std::size_t rows = matrix.size();
    std::size_t columns1 = matrix[0].size();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns1), sizeof(columns1));
    for (std::size_t i = 0; i < rows;i++){
        file.write(reinterpret_cast<const char*>(matrix[i].data()), columns1 * sizeof(double));
    }
}

void Dataset::load_binary(const std::string& file_name){
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Не удалось открыть файл для чтения: " << file_name <<std::endl;
        return;
    }
    std::size_t rows;
    std::size_t columns1;
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns1), sizeof(columns1));
    matrix.resize(rows);
    for (std::size_t i = 0;i < rows;i++){
        matrix[i].resize(columns1);
        file.read(reinterpret_cast<char*>(matrix[i].data()), columns1 * sizeof(double));
    }
}

//tests

const std::vector<std::vector<double>>& Dataset::get_matrix() const{
    return matrix;
}

//

const Column& Dataset::get_columns(int x) const {
    if (x < 0 || static_cast<std::size_t>(x) >= column_order.size()) {
        throw std::out_of_range("get_columns: index out of range");
    }
    return columns.at(column_order[x]);
}

//

double Dataset::quantile(const std::vector<double>& sorted, double q){
    if (sorted.empty()){
        throw std::invalid_argument("Quantile:Массив пустой!");
    }
    if (q < 0.0 || q > 1.0){
        throw std::invalid_argument("Quantile:Неверно указан квантиль! [0;1]");
    }

    double pos = q * (sorted.size() - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    std::size_t hi = lo + 1;

    if (hi >= sorted.size()){
        return sorted.back();
    }

    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

//

std::size_t Dataset::get_missing_count(int x) const {
    if (x < 0 || static_cast<std::size_t>(x) >= column_order.size()){
        throw std::out_of_range("get_missing_count: index out of range");
    }

    const Column& col = columns.at(column_order[x]);
    std::size_t null_count = 0;

    if (std::holds_alternative<NumericColumn>(col)){
        const auto& num = std::get<NumericColumn>(col);
        for (std::size_t i = 0; i < num.values.size(); i++){
            if (!num.values[i].has_value()){
                null_count++;
            }
        }
    }
    else{
        const auto& cat = std::get<CategoricalColumn>(col);
        for (std::size_t i = 0; i < cat.values.size(); i++){
            if (!cat.values[i].has_value()){
                null_count++;
            }
        }
    }

    return null_count;
}

//

const std::vector<std::string>& Dataset::get_matrix_column_names() const {
    return matrix_column_names;
}