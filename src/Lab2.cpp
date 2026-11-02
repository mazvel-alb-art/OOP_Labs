#include "Dataset.hpp"
#include "Matrix.hpp"
#include "DecisionStump.hpp"
#include "Exceptions.hpp"

#include <iostream>
#include <vector>
#include <span>
#include <string>

int main(){
    try {
        Dataset dataset("data/crx.csv");
        dataset.print_summary();

        
        Matrix& X = dataset.preprocess();
        std::cout << "Матрица признаков: " << X.rows() << " x " << X.cols() << "\n\n";

        
        X.save_binary("data/binary.bin");
        Matrix loaded;
        loaded.load_binary("data/binary.bin");
        std::cout << "Бинарник: сохранено " << X.rows() << " x " << X.cols()
                  << ", загружено " << loaded.rows() << " x " << loaded.cols()
                  << "\n\n";

       
        std::vector<double> y = dataset.get_target_encoded();
        Split split = dataset.train_test_split(0.8);

        std::cout << "Train: " << split.train_X.rows() << " объектов\n";
        std::cout << "Test:  " << split.test_X.rows() << " объектов\n\n";

        
        DecisionStump stump(SplitCriterion::Gini, 5);

        
        IModel& model = stump;
        model.fit(split.train_X, std::span<const double>(split.train_y));

       
        std::cout << "=== Решающий пень (Gini) ===\n";
        if (!stump.has_split()){
            std::cout << "  Разбиения не найдено (все признаки константы)\n";
            std::cout << "  Предсказание — константа: " << stump.left_value() << "\n\n";
        }
        else {
            std::cout << "  Признак:   " << stump.split_feature() << "\n";
            std::cout << "  Порог:     " << stump.split_threshold() << "\n";
            std::cout << "  left_value:  " << stump.left_value() << "\n";
            std::cout << "  right_value: " << stump.right_value() << "\n\n";
        }

        //  Предсказание
        std::vector<double> preds = model.predict(split.test_X);

        std::cout << "  Первые 10 предсказаний (test):\n";
        std::size_t n_show = std::min<std::size_t>(130, preds.size());
        for (std::size_t i = 0; i < n_show; i++){
            std::cout << "    pred = " << preds[i]
                      << "   actual = " << split.test_y[i] << "\n";
        }

       
        std::size_t correct = 0;
        for (std::size_t i = 0; i < preds.size(); i++){
            if (preds[i] == split.test_y[i]) correct++;
        }
        double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(preds.size());
        std::cout << "\n  Точность на test: " << correct << " / " << preds.size()
                  << " (" << accuracy << "%)\n";
    }
    catch (const std::exception& e){
        std::cout << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}