#ifndef GPTLANGUAGEMODEL_HPP
#define GPTLANGUAGEMODEL_HPP

#include <vector>
#include <random>
#include "./attentionmechanism.hpp"

using namespace std;

class GPTLanguageModel
{
public:
    GPTLanguageModel(int vocab_size, int n_embd, int block_size, int n_layer, int n_head);

    pair<vector<vector<vector<double>>>, double> forward(const vector<vector<int>> &idx, const vector<vector<int>> *targets = nullptr);

    vector<vector<int>> generate(vector<vector<int>> &idx, int max_new_tokens);

    //void backwards(const vector<double>& inputs, const vector<double>& targets, double learningRate);

private:
    int vocab_size;
    int n_embd;
    int block_size;
    vector<vector<double>> token_embedding_table;
    vector<vector<double>> position_embedding_table;
    vector<Block> blocks;
    LayerNorm ln_f;
    Linear lm_head;

    void initialize_weights();

    //double error(double x);
    
    //double errorDerivative(double x);

    double cross_entropy(const vector<double> &logits, const vector<double> &targets);

    vector<vector<double>> softmax(const vector<vector<double>> &logits);

    vector<vector<int>> multinomial(const vector<vector<double>> &probs, int num_samples);
};

#endif // GPTLANGUAGEMODEL_HPP