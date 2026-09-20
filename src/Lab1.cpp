#include <iostream>
#include <fstream> //для ifstream
#include <string>
#include <sstream>

#include "../include/Dataset.hpp"

int main(){
 
    Dataset dataset("data/crx.csv");
    dataset.print_summary();

}

