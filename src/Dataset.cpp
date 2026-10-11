#include <iostream>
#include <variant> 
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>


#include "Dataset.hpp"
#include "NumericColumn.hpp"
#include "CategoricalColumn.hpp"
#include "FeatureInfo.hpp"
#include "Matrix.hpp"

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
    row_count_ = count;

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
    DatasetInfo info = analyze();

    std::cout << "Объектов: " << row_count_ << std::endl;
    std::cout << "Признаков: " << column_order.size() << std::endl;
    std::cout << "#   Признак       Тип             Пропуски" << std::endl;

    std::size_t all_null_count = 0;
    std::size_t index = 0;

    for (const auto& key : column_order){
        const FeatureInfo& fi = info.at(key);
        std::cout << index << "  " << key << "            ";

        std::size_t missing = 0;
        if (std::holds_alternative<NumericFeatureInfo>(fi)){
            std::cout << "Числовой        ";
            missing = std::get<NumericFeatureInfo>(fi).missing_count;
        }
        else{
            std::cout << "Категориальный  ";
            missing = std::get<CategoricalFeatureInfo>(fi).missing_count;
        }

        double procent = static_cast<double>(missing) / row_count_ * 100;
        std::cout << missing << " (" << procent << "%)" << std::endl;

        all_null_count += missing;
        index++;
    }
    std::cout << "Всего пропусков: " << all_null_count
              << " в " << column_order.size() << " признаках" << std::endl;
}







//statistic()

void Dataset::statistics(int x){
    if (x < 0 || static_cast<std::size_t>(x) >= column_order.size()){
        return;
    }

    DatasetInfo info = analyze();
    const std::string& key = column_order[x];
    const FeatureInfo& fi = info.at(key);

    if (std::holds_alternative<NumericFeatureInfo>(fi)){
        const auto& n = std::get<NumericFeatureInfo>(fi);

        std::cout << "Признак " << x << ": " << key << " числовой" << std::endl;
        std::cout << "Пропусков: " << n.missing_count << std::endl;
        std::cout << "min:   " << n.min << "   max:    " << n.max << std::endl;
        std::cout << "mean:  " << n.mean << "   var:    " << n.var << std::endl << std::endl;
        std::cout << "q05:  " << n.q05 << std::endl;
        std::cout << "q25:  " << n.q25 << std::endl;
        std::cout << "q50:  " << n.median << std::endl;
        std::cout << "q75:  " << n.q75 << std::endl;
        std::cout << "q95:  " << n.q95 << std::endl << std::endl;

        // гистограмма — как было, через intervals
        NumericColumn* nume = std::get_if<NumericColumn>(&columns[key]);
        std::vector<std::size_t> bins;
        intervals(nume, bins);

        // min/max для подписей
        double mn = 0.0, mx = 0.0;
        bool start = false;
        for (std::size_t i = 0; i < row_count_; i++){
            if (nume->values[i].has_value()){
                if (!start){ mn = nume->values[i].value(); mx = mn; start = true; }
                else{
                    if (nume->values[i].value() < mn) mn = nume->values[i].value();
                    if (nume->values[i].value() > mx) mx = nume->values[i].value();
                }
            }
        }
        double step = (mx - mn) / static_cast<double>(NUM_BINS);
        if (step == 0.0) step = 1.0;  // защита от деления на ноль в подписях

        for (std::size_t b = 0; b < NUM_BINS; b++){
            double lo = mn + step * static_cast<double>(b);
            double hi = mn + step * static_cast<double>(b + 1);
            std::cout << "[" << lo << ", " << hi << ")    ";
            for (std::size_t j = 0; j < bins[b] / 10 + 1; j++) std::cout << "#";
            std::cout << "    " << bins[b] << std::endl;
        }
    }
    else{
        const auto& c = std::get<CategoricalFeatureInfo>(fi);

        std::cout << "Признак " << x << ": " << key << " категориальный" << std::endl;
        std::cout << "Пропусков: " << c.missing_count << std::endl;

        for (const auto& cat : c.categories){
            std::cout << cat << "    ";
            std::size_t freq = c.frequencies.at(cat);
            for (std::size_t j = 0; j < freq / 10; j++) std::cout << "#";
            std::cout << "   " << freq << std::endl;
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
    row_count_ = max_i - 2;
    return correct;

    }    
}

// 

std::size_t Dataset::null_count(NumericColumn* nume){
    std::size_t null_count = 0;
    for (std::size_t i = 0;i < row_count_;i++){
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
    for (std::size_t i = 0; i < row_count_;i++){

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
    mean = mean/(row_count_ - null_count);   
}

//

void Dataset::variance(NumericColumn* nume, double& var, double& mean, int& null_count){
    for (std::size_t i = 0; i < row_count_; i++){
        if (nume->values[i].has_value()){
            var += (nume->values[i].value() - mean) * (nume->values[i].value() - mean);
        }
    }
    var = var / (row_count_ - null_count);
}

//

std::vector<double> Dataset::q(NumericColumn* nume, const std::vector<double>& qs){
    std::vector<double> sort;
    for (std::size_t i = 0; i < row_count_; i++){
        if (nume->values[i].has_value()){
            sort.push_back(nume->values[i].value());
        }
    }
    for (std::size_t i = 0; i < sort.size()-1; i++){
        for (std::size_t j = 0; j < sort.size()-1; j++){
            if (sort[j] > sort[j+1]){
                double buffer = sort[j];
                sort[j] = sort[j+1];
                sort[j+1] = buffer;
            }
        }
    }

    std::vector<double> result;
    result.reserve(qs.size());

    for (std::size_t k = 0; k < qs.size(); k++){
        double q = qs[k];
        double pos = q * (sort.size() - 1);
        std::size_t lo = static_cast<std::size_t>(pos);
        std::size_t hi = lo + 1;

        if (hi >= sort.size()){
            result.push_back(sort.back());
        }
        else{
            double frac = pos - lo;
            result.push_back(sort[lo] * (1.0 - frac) + sort[hi] * frac);
        }
    }

    return result;
}

//

void Dataset::intervals(NumericColumn* nume, std::vector<std::size_t>& bins){
    bins.assign(NUM_BINS, 0);

    // находим min и max по значениям
    double min = 0.0;
    double max = 0.0;
    bool start = false;
    for (std::size_t i = 0; i < row_count_; i++){
        if (nume->values[i].has_value()){
            if (!start){
                min = nume->values[i].value();
                max = nume->values[i].value();
                start = true;
            }
            else{
                if (nume->values[i].value() < min) min = nume->values[i].value();
                if (nume->values[i].value() > max) max = nume->values[i].value();
            }
        }
    }

    if (!start){
        return;  // все значения — пропуски
    }

    double range = max - min;
    if (range == 0.0){
        // все значения одинаковы — кладём всё в первый бин
        bins[0] = 1;
        for (std::size_t i = 1; i < row_count_; i++){
            if (nume->values[i].has_value()) bins[0]++;
        }
        return;
    }

    double step = range / static_cast<double>(NUM_BINS);

    for (std::size_t i = 0; i < row_count_; i++){
        if (nume->values[i].has_value()){
            double v = nume->values[i].value();
            std::size_t idx = static_cast<std::size_t>((v - min) / step);
            if (idx >= NUM_BINS) idx = NUM_BINS - 1;  // max попадает в последний бин
            bins[idx]++;
        }
    }
}

//

std::size_t Dataset::null_count(CategoricalColumn* cate){
    std::size_t null_count = 0;
    for (std::size_t i = 0; i < row_count_;i++){
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
    for (std::size_t i = 0; i < row_count_;i++){      //заполнение списка уникальных категорий
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
        for (std::size_t j = 0;j < row_count_;j++){
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

std::vector<double> Dataset::q50()  {
    std::vector<double> medians(column_order.size(), 0.0);
    NumericColumn* nume;
    double q50;
    for (std::size_t k = 0; k < column_order.size(); k++){
        std::string key = column_order[k];
        nume = std::get_if<NumericColumn>(&columns.at(key));
        if (nume != nullptr){
            std::vector<double> sort;
            for (std::size_t i = 0; i < row_count_; i++){
                if (nume->values[i].has_value()){
                    sort.push_back(nume->values[i].value());
                }
            }
            for (std::size_t i = 0; i < sort.size()-1; i++){
                for (std::size_t j = 0; j < sort.size()-1; j++){
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
            medians[k] = q50;
        }
    }
    return medians;
}

DatasetInfo Dataset::analyze() {
    DatasetInfo info;

    for (const auto& name : column_order){
        const Column& col = columns.at(name);

        if (std::holds_alternative<NumericColumn>(col)){
            const auto& num = std::get<NumericColumn>(col);
            NumericFeatureInfo ni;

            // пропуски
            for (std::size_t i = 0; i < row_count_; i++){
                if (!num.values[i].has_value()){
                    ni.missing_count++;
                }
            }

            // min, max, mean
            double mean = 0.0;
            int dummy_null = static_cast<int>(ni.missing_count);
            min_max_mean(const_cast<NumericColumn*>(&num), ni.min, ni.max, mean, dummy_null);

            // variance
            double var = 0.0;
            variance(const_cast<NumericColumn*>(&num), var, mean, dummy_null);

            ni.mean = mean;
            ni.var = var;

            // квантили через существующий q()
            std::vector<double> qs = {0.05, 0.25, 0.50, 0.75, 0.95};
            std::vector<double> res = q(const_cast<NumericColumn*>(&num), qs);

            ni.q05 = res[0];
            ni.q25 = res[1];
            ni.median = res[2];
            ni.q75 = res[3];
            ni.q95 = res[4];

            info[name] = ni;
        }
        else{
            const auto& cat = std::get<CategoricalColumn>(col);
            CategoricalFeatureInfo ci;

            for (std::size_t i = 0; i < row_count_; i++){
                if (!cat.values[i].has_value()){
                    ci.missing_count++;
                    continue;
                }
                const std::string& c = cat.values[i].value();

                bool found = false;
                for (const auto& existing : ci.categories){
                    if (existing == c){ found = true; break; }
                }
                if (!found){
                    ci.categories.push_back(c);
                }
                ci.frequencies[c]++;
            }

            info[name] = ci;
        }
    }
    return info;
}

Matrix& Dataset::preprocess(){
    matrix_.get_matrix_column_names().clear();
    matrix_.data().clear();

    CategoricalColumn* cate;
    NumericColumn* nume;
    std::vector<double> medians = Dataset::q50();
    std::vector<std::vector<std::string>> category_list(column_order.size());

    for (std::size_t i = 0; i < column_order.size(); i++){
        std::string key = column_order[i];
        auto& column = columns[key];
        nume = std::get_if<NumericColumn>(&column);
        if (nume != nullptr){
        }
        else {
            cate = std::get_if<CategoricalColumn>(&column);
            bool in = false;
            for (std::size_t j = 0; j < row_count_; j++){
                if (cate->values[j].has_value()){
                    for (std::size_t l = 0; l < category_list[i].size(); l++){
                        if (cate->values[j].value() == category_list[i][l]){
                            in = true;
                        }
                    }
                    if (in == false){
                        category_list[i].push_back(cate->values[j].value());
                    }
                }
                else{
                    for (std::size_t l = 0; l < category_list[i].size(); l++){
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

    std::vector<std::vector<double>> result(row_count_);
    std::vector<std::vector<std::variant<std::string, double>>> filled(row_count_);

    for (std::size_t i = 0; i < row_count_; i++){
        for (std::size_t j = 0; j < column_order.size(); j++){
            std::string key = column_order[j];
            nume = std::get_if<NumericColumn>(&columns[key]);

            if (nume != nullptr){
                if (nume->values[i].has_value()){
                    filled[i].push_back(nume->values[i].value());
                }
                else{
                    filled[i].push_back(medians[j]);
                }
            }
            else{
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

    std::vector<double> mass_min(column_order.size());
    std::vector<double> mass_max(column_order.size());

    {
        for (std::size_t i = 0; i < column_order.size(); i++){
            double min;
            double max;
            bool start = false;
            std::string key = column_order[i];
            nume = std::get_if<NumericColumn>(&columns[key]);
            if (nume != nullptr){
                for (std::size_t j = 0; j < row_count_; j++){
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

    for (std::size_t i = 0; i < column_order.size(); i++){
        std::string key = column_order[i];
        if (key == target_name_) continue;
        nume = std::get_if<NumericColumn>(&columns[key]);

        if (nume != nullptr){
            matrix_.get_matrix_column_names().push_back(key);
            for (std::size_t j = 0; j < row_count_; j++){
                double scaled = ((std::get<double>(filled[j][i])) - mass_min[i]) / (mass_max[i] - mass_min[i]);
                result[j].push_back(scaled);
            }
        }
        else{
            for (std::size_t k = 0; k < category_list[i].size(); k++){
                matrix_.get_matrix_column_names().push_back(key + "_" + category_list[i][k]);
                for (std::size_t j = 0; j < row_count_; j++){
                    if (category_list[i][k] == (std::get<std::string>(filled[j][i]))){
                        result[j].push_back(1);
                    }
                    else{
                        result[j].push_back(0);
                    }
                }
            }
        }
    }

    std::size_t cols = result[0].size();
    matrix_.set_size(row_count_, cols);

    for (std::size_t i = 0; i < row_count_; i++){
        for (std::size_t j = 0; j < cols; j++){
            matrix_.data().push_back(result[i][j]);
        }
    }

    return matrix_;
}




//tests

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

