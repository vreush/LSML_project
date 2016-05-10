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


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sign(double x) {
    if (x >= 0) {
        return 1;
    } else {
        return -1;
    }
}


double predict_sample_proba(const std::unordered_map<std::string,double>& sample,
               const std::unordered_map<std::string,double>& weights) {
    double pred = 0;
    for (auto kv : sample) {
        auto found = weights.find(kv.first);
        double coef = 0;
        if (found != weights.end()) {
            coef = found->second;
        }
        pred += coef * kv.second;
    }
    return sigmoid(pred);
}


std::vector<double> predict_samples_proba(const std::vector<std::unordered_map<std::string, double>>& samples,
                            const std::unordered_map<std::string,double>& weights) {
    std::vector<double> result;
    result.reserve(samples.size());
    for (const auto& sample : samples) {
        result.push_back(predict_sample_proba(sample, weights));
    }
    return result;
}


double log_loss(const std::vector<double>& y_pred, const std::vector<int>& y_true) {
    double loss = 0;
    for (size_t index = 0; index < y_pred.size(); ++index) {
        loss += y_true[index] * log(y_pred[index]) + ((1.0 - y_true[index]) * (log(1.0 - y_pred[index])));
    }
    return -loss / y_pred.size();
}


// ftrl
struct Params {
    double alpha;
    double beta;
    double lambda1;
    double lambda2;
    std::unordered_map<std::string,double> z; 
    std::unordered_map<std::string,double> n;
    std::unordered_map<std::string,double> g;
    std::unordered_map<std::string,double> weights;
    std::unordered_map<std::string,double> sigma;
};


std::unordered_map<std::string,double> run_ftrl(const std::vector<std::unordered_map<std::string,double>>& samples,
               const std::vector<int> labels,
               Params &params) {

    for (size_t index = 0; index < samples.size(); ++index) {
        const auto& sample = samples[index];
        int label = labels[index];
        for(const auto& item: sample) {
             auto feature = item.first;
             auto val = item.second;
             params.weights[feature] = 0;
             auto eta = (1.0 / (
                     (params.beta + sqrt(params.n[feature]) / params.alpha
                         )
                     + params.lambda2));
             if (params.z[feature] >= params.lambda1) {
                 params.weights[feature] = - eta * (params.z[feature] - (sign(params.z[feature]) * params.lambda1));
             }

             double pt = predict_sample_proba(sample, params.weights);

             params.g[feature] = (pt - label) * sample.at(feature);
             params.sigma[feature] = (sqrt(params.n[feature] + (params.g[feature] * params.g[feature])) - sqrt(params.n[feature])) / params.alpha;
             params.z[feature] = params.z[feature] + params.g[feature] - (params.sigma[feature] * params.weights[feature]);
             params.n[feature] = params.n[feature] + (params.g[feature] * params.g[feature]);
        }
    }
    return params.weights;
}


std::unordered_map<std::string,double> batch_train_ftrl(std::ifstream& file, size_t batch_size,
    double alpha, double beta, double lambda1, double lambda2) {

    std::vector<std::unordered_map<std::string,double>> features;
    vector<int> ground_truth;

    features.reserve(batch_size);
    ground_truth.reserve(batch_size);

    string line;
    size_t count = 0;
    Params params;
    params.alpha = alpha;
    params.beta = beta;
    params.lambda1 = lambda1;
    params.lambda2 = lambda2;

    std::unordered_map<std::string,double> weights;
    while (file >> line) {
        count += 1;
        std::unordered_map<std::string,double> row;
        bool is_first = true;
        for(std::string el : split(line, ',')) {
            if (!is_first) {
                auto name_val = split(el, ':');
                double number = std::stod(name_val[1]);
                row[name_val[0]] = number;
            } else {
                ground_truth.push_back(std::stoi(el));
                is_first = false;
            }
        }
        features.push_back(row);
        if ((count % 10000) == 0) {
            std::cout << "count = " << count << "\n";            
        }
        if (count >= batch_size) {
            std::cout << "count = " << count << "\n";
            weights = run_ftrl(features, ground_truth, params);
            
            count = 0;
            
            features.clear();
            ground_truth.clear();
            
            features.reserve(batch_size);
            ground_truth.reserve(batch_size);
        }
    }
    if (count > 0) {
            weights = run_ftrl(features, ground_truth, params);
    }
    std::cout << "training done--------------------------------------------------\n";
    return weights;
}


void export_data(const std::vector<double>& preds, std::ofstream& out) {
    for (const auto& pred: preds) {
        out << pred << "\n";
    }
}


void batch_predict_ftrl(std::ifstream& file, std::ofstream& out, size_t batch_size,
    const std::unordered_map<std::string,double>& weights) {
    std::vector<std::unordered_map<std::string,double>> features;
    vector<int> ground_truth;
    vector<double> preds;

    features.reserve(batch_size);
    ground_truth.reserve(batch_size);

    string line;
    size_t count = 0;
    Params params;
    while (file >> line) {
        count += 1;
        std::unordered_map<std::string,double> row;
        bool is_first = true;
        for(std::string el : split(line, ',')) {
            if (!is_first) {
                auto name_val = split(el, ':');
                double number = std::stod(name_val[1]);
                row[name_val[0]] = number;
            } else {
                ground_truth.push_back(std::stoi(el));
                is_first = false;
            }
        }
        features.push_back(row);
        if ((count % 10000) == 0) {
            std::cout << "count = " << count << "\n";            
        }
        if (count >= batch_size) {
            vector<double> batch_preds = predict_samples_proba(features, weights);
            preds.insert(preds.end(), batch_preds.begin(), batch_preds.end());
            
            count = 0;
            
            features.clear();            
            features.reserve(batch_size);
        }
    }
    if (count > 0) {
        vector<double> batch_preds = predict_samples_proba(features, weights);
        preds.insert(preds.end(), batch_preds.begin(), batch_preds.end());
    }
    std::cout << "----- test log loss: " << log_loss(preds, ground_truth) << "\n";
    std::cout << "--------------------------------------------------\n";
    export_data(preds, out);
}


int main(int argc, char* argv[]) {
    std::string filename_train = argv[1];
    std::string filename_test = argv[2];
    std::string filename_out = argv[3];
    size_t batch_size = std::stoi(argv[4]);
    double alpha = std::stod(argv[5]);
    double beta = std::stod(argv[6]);
    double lambda1 = std::stod(argv[7]);
    double lambda2 = std::stod(argv[8]);

    std::cout << "train: " << filename_train << "\n";
    std::cout << "test: " << filename_test << "\n";
    std::cout << "out: " << filename_out << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "alpha: " << alpha << "\n";
    std::cout << "beta: " << beta << "\n";
    std::cout << "lambda1: " << lambda1 << "\n";
    std::cout << "lambda2: " << lambda2 << "\n";

    ifstream input_train;
    input_train.open(filename_train);
    std::unordered_map<std::string,double> model_weights = batch_train_ftrl(input_train, batch_size, alpha, beta, lambda1, lambda2);
    input_train.close();

    ifstream input_test;
    input_test.open(filename_test);
    ofstream out;
    out.open(filename_out);

    batch_predict_ftrl(input_test, out, batch_size, model_weights);
    input_test.close();

    return 0;
}
