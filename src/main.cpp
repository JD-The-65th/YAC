
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "Yet Another Calculator (For Steam Deck Arrival Dates)" << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/iris_plant_original.csv", ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 3;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<type, 2> inputs(3, 4);

        inputs.setValues({{type(5.1),type(3.5),type(1.4),type(0.2)},
                          {type(6.4),type(3.2),type(4.5),type(1.5)},
                          {type(6.3),type(2.7),type(4.9),type(1.8)}});

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        cout << "\nConfusion matrix:\n" << confusion << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_c("../data/neural_network.c");
        neural_network.save_expression_python("../data/neural_network.py");

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}  