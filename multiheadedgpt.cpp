#include "./multiheadedgpt.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

GPTLanguageModel::GPTLanguageModel(int vocab_size, int n_embd, int block_size, int n_layer, int n_head)
    : vocab_size(vocab_size), n_embd(n_embd), block_size(block_size),
      token_embedding_table(vocab_size, vector<double>(n_embd)),
      position_embedding_table(block_size, vector<double>(n_embd)),
      blocks(vector<Block>(n_layer, Block(n_embd, n_head))),
      ln_f(LayerNorm(n_embd)),
      lm_head(Linear(n_embd, vocab_size))
{
    initialize_weights();
}

pair<vector<vector<vector<double>>>, double> GPTLanguageModel::forward(const vector<vector<int>> &idx, const vector<vector<int>> *targets)
{
    int B = idx.size();
    int T = idx[0].size();

    vector<vector<double>> tok_emb(B, vector<double>(T, 0.0));
    // Get the embedding of the specific token
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            tok_emb[i][j] = token_embedding_table[0][j];
        }
    }

    vector<vector<double>> pos_emb(T, vector<double>(n_embd));
    for (int i = 0; i < T; ++i)
    {
        pos_emb[i] = position_embedding_table[i];
    }

    vector<vector<vector<double>>> x(B, vector<vector<double>>(T, vector<double>(n_embd)));
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            x[0][i][j] = tok_emb[i][j] + pos_emb[i][j];
        }
    }

    for (int i = 0; i < blocks.size(); ++i)
    {
        x = blocks[i].forward(x);
    }

    x = ln_f.forward(x);
    vector<vector<vector<double>>> logits = lm_head.forward(x);

    double loss = 0.0;
    if (targets != nullptr)
    {
        int B = logits.size();
        int T = logits[0].size();
        int C = logits[0][0].size();
        vector<double> flat_logits;
        vector<double> flat_targets;
        for (int i = 0; i < B; ++i)
        {
            for (int j = 0; j < T; ++j)
            {
                for (int k = 0; k < C; ++k)
                {
                    flat_logits.push_back(logits[i][j][k]);
                    flat_targets.push_back((*targets)[i][j]);
                }
            }
        }
        loss = cross_entropy(flat_logits, flat_targets);
    }

    return make_pair(logits, loss);
}

vector<vector<int>> GPTLanguageModel::generate(vector<vector<int>> &idx, int max_new_tokens)
{
    for (int i = 0; i < max_new_tokens; ++i)
    {
        vector<vector<int>> idx_cond;
        for (auto &seq : idx)
        {
            idx_cond.push_back(vector<int>(seq.end() - block_size, seq.end()));
        }
        auto [logits, loss] = forward(idx_cond);
        vector<vector<double>> cropped_logits = vector<vector<double>>(logits.size(), vector<double>(logits[0].size()));
        for (int j = 0; j < logits.size(); ++j)
        {
            cropped_logits[j] = logits[0][j];
        }
        vector<vector<double>> probs = softmax(cropped_logits);
        vector<vector<int>> idx_next = multinomial(probs, 1);
        for (int j = 0; j < idx.size(); ++j)
        {
            idx[j].push_back(idx_next[j][0]);
        }

        // Print attention vectors 
        double total = 0.0;
        for (const auto &attention_vector : logits)
        {
            for (const auto &val : attention_vector[0])
            {
                total += val;
            }
        }

        cout << "Generated token... " ;
    }
    cout << endl;
    return idx;
}

void GPTLanguageModel::initialize_weights()
{
    random_device rd;
    mt19937 gen(42);
    normal_distribution<> d(0.0, 0.02);
    for (auto &row : token_embedding_table)
    {
        std::generate(row.begin(), row.end(), [&]()
                 { return d(gen); });
    }
    for (auto &row : position_embedding_table)
    {
        std::generate(row.begin(), row.end(), [&]()
                 { return d(gen); });
    }

    cout << "Initialized weights" << endl;
    cout << "Token embedding table: " << token_embedding_table.size() << " x " << token_embedding_table[0].size() << endl;
    cout << "Position embedding table: " << position_embedding_table.size() << " x " << position_embedding_table[0].size() << endl;
    cout << "Blocks: " << blocks.size() << endl;
}

double GPTLanguageModel::cross_entropy(const vector<double> &logits, const vector<double> &targets)
{
    int count = logits.size();
    double loss = 0.0;
    for (int i = 0; i < count; ++i) {
        loss -= logits[i] * (targets[i] - (logits[i] >= 0)) -
            log(1 + exp(logits[i] - 2 * logits[i] * (logits[i] >= 0)));
      }


    return loss;
}

vector<vector<double>> GPTLanguageModel::softmax(const vector<vector<double>> &logits)
{
    vector<vector<double>> probs(logits.size(), vector<double>(logits[0].size(), 0.0));
    for (int i = 0; i < logits.size(); ++i)
    {
        double max_val = *max_element(logits[i].begin(), logits[i].end());
        double sum = 0.0;
        for (int j = 0; j < logits[0].size(); ++j)
        {
            probs[i][j] = exp(logits[i][j] - max_val);
            sum += probs[i][j];
        }
        for (int j = 0; j < logits[0].size(); ++j)
        {
            probs[i][j] /= sum;
        }
    }
    return logits;
}

vector<vector<int>> GPTLanguageModel::multinomial(const vector<vector<double>> &probs, int num_samples)
{
    vector<vector<int>> rslt(probs.size(), vector<int>(num_samples, 0));
    for (int i = 0; i < probs.size(); ++i)
    {
        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> d(probs[i].begin(), probs[i].end());
        for (int j = 0; j < num_samples; ++j)
        {
            rslt[i][j] = d(gen);
        }
    }
    return rslt;
}