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

    vector<vector<vector<double>>> tok_emb(B, vector<vector<double>>(T, vector<double>(n_embd, 0.0)));
    // Get the embedding of the specific token
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            int token_idx = idx[i][j];
            if (token_idx >= 0 && token_idx < vocab_size) {
                tok_emb[i][j] = token_embedding_table[token_idx];
            }
        }
    }

    vector<vector<double>> pos_emb(T, vector<double>(n_embd));
    #pragma omp parallel for
    for (int i = 0; i < T; ++i)
    {
        pos_emb[i] = position_embedding_table[i];
    }

    vector<vector<vector<double>>> x(B, vector<vector<double>>(T, vector<double>(n_embd)));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            for (int k = 0; k < n_embd; ++k)
            {
                x[i][j][k] = tok_emb[i][j][k] + pos_emb[j][k];
            }
        }
    }

    // Can't be parallelized
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
        #pragma omp parallel for
        for (auto &seq : idx)
        {
            idx_cond.push_back(vector<int>(seq.end() - block_size, seq.end()));
        }
        auto [logits, loss] = forward(idx_cond);
        vector<vector<double>> cropped_logits = vector<vector<double>>(logits.size(), vector<double>(logits[0].size()));
        #pragma omp parallel for
        for (int j = 0; j < logits.size(); ++j)
        {
            cropped_logits[j] = logits[0][j];
        }
        vector<vector<double>> probs = softmax(cropped_logits);
        vector<vector<int>> idx_next = multinomial(probs, 1);
        #pragma omp parallel for
        for (int j = 0; j < idx.size(); ++j)
        {
            idx[j].push_back(idx_next[j][0]);
        }

        cout << "Generated token... " ;
    }
    cout << endl;
    return idx;
}

void GPTLanguageModel::backward(const vector<vector<int>> &idx, const vector<vector<int>> &targets, double learning_rate)
{
    // Initialize gradients if needed
    if (token_embedding_grads.empty()) {
        token_embedding_grads.resize(vocab_size, vector<double>(n_embd, 0.0));
        position_embedding_grads.resize(block_size, vector<double>(n_embd, 0.0));
    }
    
    // Zero out gradients
    for (auto &row : token_embedding_grads) {
        fill(row.begin(), row.end(), 0.0);
    }
    for (auto &row : position_embedding_grads) {
        fill(row.begin(), row.end(), 0.0);
    }
    
    // Forward pass to get logits and loss (single forward pass)
    auto [logits, loss] = forward(idx, &targets);
    
    int B = idx.size();
    int T = idx[0].size();
    int C = vocab_size;
    
    // Compute softmax probabilities once for backprop
    vector<vector<vector<double>>> probs(B, vector<vector<double>>(T, vector<double>(C, 0.0)));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            double max_val = *max_element(logits[b][t].begin(), logits[b][t].end());
            double sum = 0.0;
            for (int c = 0; c < C; ++c) {
                probs[b][t][c] = exp(logits[b][t][c] - max_val);
                sum += probs[b][t][c];
            }
            for (int c = 0; c < C; ++c) {
                probs[b][t][c] /= sum;
            }
        }
    }
    
    // Compute gradients: pred - target (cross-entropy + softmax gradient)
    vector<vector<vector<double>>> dlogits(B, vector<vector<double>>(T, vector<double>(C, 0.0)));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int target_idx = targets[b][t];
            for (int c = 0; c < C; ++c) {
                dlogits[b][t][c] = probs[b][t][c] / (B * T);
            }
            if (target_idx >= 0 && target_idx < C) {
                dlogits[b][t][target_idx] -= 1.0 / (B * T);
            }
        }
    }
    
    // Backprop through embeddings
    // Simplified: update token embeddings with gradient from output
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int token_idx = idx[b][t];
            if (token_idx >= 0 && token_idx < vocab_size) {
                // Accumulate gradients
                for (int e = 0; e < n_embd; ++e) {
                    // Approximate gradient propagation to embeddings
                    double grad = 0.0;
                    for (int c = 0; c < C; ++c) {
                        grad += dlogits[b][t][c];
                    }
                    token_embedding_grads[token_idx][e] += grad / C;
                }
            }
            
            // Position embedding gradients
            if (t < block_size) {
                for (int e = 0; e < n_embd; ++e) {
                    double grad = 0.0;
                    for (int c = 0; c < C; ++c) {
                        grad += dlogits[b][t][c];
                    }
                    position_embedding_grads[t][e] += grad / C;
                }
            }
        }
    }
    
    // Update embeddings with gradients
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            token_embedding_table[i][j] -= learning_rate * token_embedding_grads[i][j];
        }
    }
    
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            position_embedding_table[i][j] -= learning_rate * position_embedding_grads[i][j];
        }
    }
    
    cout << "Backward pass complete. Loss: " << loss << endl;
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
    cout << "Token embedding table: " << token_embedding_table.size();
    if (token_embedding_table.size() > 0) {
        cout << " x " << token_embedding_table[0].size();
    }
    cout << endl;
    cout << "Position embedding table: " << position_embedding_table.size();
    if (position_embedding_table.size() > 0) {
        cout << " x " << position_embedding_table[0].size();
    }
    cout << endl;
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