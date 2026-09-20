#include <iostream>
#include <fstream> //для ifstream
#include <string>
#include <sstream>

#include "../include/Dataset.hpp"

int main(){
 
    Dataset dataset("data/crx.csv");
    dataset.print_summary();
    std::cout<<"Введите номер признака, статистику о коором желаете получить: "<<std::endl;
    int x;
    std::cin>>x;
    dataset.statistics(x);
}

