#include<string>
#include<iostream>
#include<sstream>
#include<vector>
#include<fstream>
#include<unordered_map>
#include<set>
#include<cmath>
#include<random>

using namespace std;


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


vector<vector<double>> read_csv(std::ifstream& file) {
    vector<vector<double>> result;
    string line;
    std::cout << "in read_csv\n";
    while (file >> line) {
        std::cout << "line read\n";
        vector<double> row;
        size_t col = 0;
        for(std::string el : split(line, ',')) {
            double number = std::stod(el);
            row.push_back(number);
            col += 1;
        }
        result.push_back(row);
    }
    return result;
}


std::vector<std::unordered_map<char,int>> read_data(std::ifstream& file, size_t lines_count) {
    std::vector<std::unordered_map<char,int>> result;
    return result;
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double predict_sample_proba(const std::unordered_map<std::string,double>& sample,
               const std::unordered_map<std::string,double>& weights) {
    double pred = 0;
    for (auto kv : sample) {
        pred += weights.find(kv.first)->second * kv.second;
    }
    return sigmoid(pred);
}

std::vector<double> predict_sample_proba(const std::vector<std::unordered_map<std::string, double>>& samples,
                            const std::unordered_map<std::string,double>& weights) {
    std::vector<double> result;
    result.reserve(samples.size());
    for (const auto& sample : samples) {
        result.push_back(predict_sample_proba(sample, weights));
    }
    return result;
}

const std::unordered_map<std::string,double> grad(const std::unordered_map<std::string,double>& sample,
                                                  double ground_truth,
                                                  const std::unordered_map<std::string,double>& weights) {
    std::unordered_map<std::string,double> result;
    double mult = (predict_sample_proba(sample, weights) - ground_truth);
    for (auto kv : sample) {
         result[kv.first] = mult * kv.second;
    }
    return result;
}

void print_data(const std::unordered_map<std::string,double>& data) {
    for (auto kv: data) {
        std::cout << kv.first << ": " << kv.second << "; ";
    }
    std::cout << "\n";
}

double log_loss(const std::vector<double>& y_pred, const std::vector<int>& y_true) {
    double loss = 0;
    for (size_t index = 0; index < y_pred.size(); ++index) {
        loss += y_true[index] * log(y_pred[index]) + ((1.0 - y_true[index]) * (log(1.0 - y_pred[index])));
    }
    return -loss;
}

void update(std::unordered_map<std::string,double >& weights,
            const std::unordered_map<std::string,double>& grad,
            double learning_rate, double lambda) {
    for (auto kv : grad) {
        weights[kv.first] -= ((learning_rate * grad.find(kv.first)->second) + (lambda * weights[kv.first]));
    }
}

void do_sgd_step(const std::unordered_map<std::string,double>& sample,
                 double ground_truth,
                 std::unordered_map<std::string,double>& weights,
                 double learning_rate,
                 double lambda) {
    auto cur_grad = grad(sample, ground_truth, weights);
    update(weights, cur_grad, learning_rate, lambda);
}


size_t get_rand_index(size_t max_index) {
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, max_index);
    return distribution(generator);
}

std::unordered_map<std::string,double> train(const std::vector<std::unordered_map<std::string,double>>& samples,
           const std::vector<int> ground_truth, double learning_rate, double lambda, unsigned int max_iter) {
    std::cout << "in train\n";
    std::cout << samples.size() << "\n";
    std::unordered_map<std::string,double> weights = samples[0];
    // std::unordered_map<std::string,double> weights = std::unordered_map<std::string,double>(samples[0]);
    std::cout << "in train2\n";
    for (auto kv: weights) {
        std::cout << kv.first << "\n";
        weights[kv.first] = 1.0;
    }
    std::cout << "init weights\n";
    print_data(weights);
    for (size_t i = 0; i < max_iter; ++i) {
        size_t index = get_rand_index(samples.size());
        auto sample = samples[index];
        double true_label = ground_truth[index];
        do_sgd_step(sample, true_label, weights, learning_rate, lambda);

        i += 1;
        std::cout << "after iter " << i << ":\n";
        print_data(weights);
        std::cout << "-----log loss: " << log_loss(predict_sample_proba(samples, weights), ground_truth) << "\n";
    }
    return weights;
}


std::vector<double> predict_proba(const std::unordered_map<std::string,double>& weights,
                                  const std::vector<std::unordered_map<std::string,double>>& samples) {
    std::vector<double> probas;
    probas.reserve(samples.size());
    for (const auto& sample: samples) {
        probas.push_back(predict_sample_proba(sample, weights));
    }
    return probas;
}

void export_data(const std::vector<double>& preds, std::ofstream& out) {
    for (const auto& pred: preds) {
        out << pred << "\n";
    }
}


struct Data {
    std::vector<std::unordered_map<std::string,double>> features;
    std::vector<int> ground_truth;
};


Data read_data(std::ifstream& input) {
    auto data = read_csv(input);

    std::vector<std::unordered_map<std::string,double>> data_tr;
    std::vector<int> ground_truth;

    for (const auto& row: data) {
        std::unordered_map<std::string,double> row_tr;
        ground_truth.push_back(row[0]);
        for (size_t i = 1; i < row.size(); ++i) {
            row_tr[std::to_string(i)] = row[i];
        }
        data_tr.push_back(row_tr);
        std::cout << "\n";
    }
    std::cout << "data len:" << data_tr.size() << "\n";

//    for (const auto& row: data_tr) {
//        for (auto kv: row) {
//            std::cout << kv.first << " ";
//            std::cout << kv.second << " ";
//        }
//        std::cout << "\n";
//    }
    return Data{data_tr, ground_truth};
}
//g++ -std=c++0x -o lr main.cpp

int main(int argc, char* argv[]) {
//    std::string filename_train = argv[1];
//    std::string filename_test = argv[2];
    std::string filename_train = "train.csv";
    std::string filename_test = "train.csv";
    std::string filename_out = "out.csv";

    std::cout << filename_train << "\n";
    std::cout << filename_test << "\n";
    std::cout << filename_out << "\n";
    std::cout << "cin\n";

    // auto dir = std::string("C:\\Users\\viktor\\ClionProjects\\inheritance\\");
    auto dir = std::string("");

//    std::cin >> filename_test;
    filename_train = dir + filename_train;
    filename_test = dir + filename_test;
    filename_out = dir + filename_out;

    std::cout << "train path: " << filename_train << "\n";

    ifstream input_train;
    input_train.open(filename_train);
    auto train_data = read_data(input_train);
    input_train.close();
    if (train_data.features.size() == 0) {
        throw std::runtime_error("No train file at " + filename_train);
    }
    


    ifstream input_test;
    input_test.open(filename_test);
    auto test_data = read_data(input_test);
    input_test.close();
    if (test_data.features.size() == 0) {
        throw std::runtime_error("No test file at " + filename_test);
    }

//    file.open("C:\\Users\\viktor\\ClionProjects\\inheritance\\train.csv");


    double learning_rate = 0.1;
    double lambda = 0;
    unsigned int max_iter = 1000;

    std::cout << "train\n";
    train_data.features;
    train_data.ground_truth;
    auto model_weights = train(train_data.features, train_data.ground_truth, learning_rate, lambda, max_iter);
    std::cout << "train end\n";

    auto test_preds = predict_proba(model_weights, test_data.features);

    ofstream out;
    out.open(filename_out);
    export_data(test_preds, out);
    out.close();

    return 0;
}
