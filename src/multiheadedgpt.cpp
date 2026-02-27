#include "./multiheadedgpt.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

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

    // Get token embeddings - optimized with parallel access
    vector<vector<vector<double>>> tok_emb(B, vector<vector<double>>(T, vector<double>(n_embd, 0.0)));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            int token_idx = idx[i][j];
            if (token_idx >= 0 && token_idx < vocab_size) {
                // Direct memory copy is faster than element-wise
                const vector<double>& embedding = token_embedding_table[token_idx];
                #pragma omp simd
                for (int k = 0; k < n_embd; ++k) {
                    tok_emb[i][j][k] = embedding[k];
                }
            }
        }
    }

    // Add position embeddings - optimized
    vector<vector<vector<double>>> x(B, vector<vector<double>>(T, vector<double>(n_embd)));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            const vector<double>& pos_emb = position_embedding_table[j];
            #pragma omp simd
            for (int k = 0; k < n_embd; ++k)
            {
                x[i][j][k] = tok_emb[i][j][k] + pos_emb[k];
            }
        }
    }

    // Forward through transformer blocks
    for (size_t i = 0; i < blocks.size(); ++i)
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
        // Crop to block_size - FIXED: no race condition
        vector<vector<int>> idx_cond;
        idx_cond.reserve(idx.size());
        for (const auto &seq : idx)
        {
            int start_pos = max(0, (int)seq.size() - block_size);
            idx_cond.push_back(vector<int>(seq.begin() + start_pos, seq.end()));
        }
        
        auto [logits, loss] = forward(idx_cond);
        
        // Get logits for last position
        vector<vector<double>> last_logits(logits.size(), vector<double>(logits[0][0].size()));
        for (size_t j = 0; j < logits.size(); ++j)
        {
            int last_t = logits[j].size() - 1;
            last_logits[j] = logits[j][last_t];
        }
        
        vector<vector<double>> probs = softmax(last_logits);
        vector<vector<int>> idx_next = multinomial(probs, 1);
        
        for (size_t j = 0; j < idx.size(); ++j)
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

void GPTLanguageModel::save_model(const string &filename)
{
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }
    
    file << "{\n";
    
    // Save token embeddings
    file << "  \"token_embedding_table\": [\n";
    for (size_t i = 0; i < token_embedding_table.size(); ++i) {
        file << "    [";
        for (size_t j = 0; j < token_embedding_table[i].size(); ++j) {
            file << token_embedding_table[i][j];
            if (j < token_embedding_table[i].size() - 1) file << ", ";
        }
        file << "]";
        if (i < token_embedding_table.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";
    
    // Save position embeddings
    file << "  \"position_embedding_table\": [\n";
    for (size_t i = 0; i < position_embedding_table.size(); ++i) {
        file << "    [";
        for (size_t j = 0; j < position_embedding_table[i].size(); ++j) {
            file << position_embedding_table[i][j];
            if (j < position_embedding_table[i].size() - 1) file << ", ";
        }
        file << "]";
        if (i < position_embedding_table.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";
    
    // Save other model parameters
    file << "  \"hyperparameters\": {\n";
    file << "    \"vocab_size\": " << vocab_size << ",\n";
    file << "    \"n_embd\": " << n_embd << ",\n";
    file << "    \"block_size\": " << block_size << "\n";
    file << "  }\n";
    
    file << "}\n";
    
    file.close();
    cout << "Model saved to " << filename << endl;
}

bool GPTLanguageModel::load_model(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << " for reading" << endl;
        return false;
    }
    
    string line;
    bool in_token_embeddings = false;
    bool in_position_embeddings = false;
    int row_idx = 0;
    
    while (getline(file, line)) {
        // Skip empty lines and braces
        if (line.find("token_embedding_table") != string::npos) {
            in_token_embeddings = true;
            in_position_embeddings = false;
            row_idx = 0;
            continue;
        }
        if (line.find("position_embedding_table") != string::npos) {
            in_token_embeddings = false;
            in_position_embeddings = true;
            row_idx = 0;
            continue;
        }
        if (line.find("hyperparameters") != string::npos) {
            break;
        }
        
        // Parse array rows
        size_t start = line.find('[');
        size_t end = line.rfind(']');
        if (start != string::npos && end != string::npos && end > start) {
            string numbers = line.substr(start + 1, end - start - 1);
            vector<double> row;
            stringstream ss(numbers);
            string num;
            
            while (getline(ss, num, ',')) {
                // Trim whitespace
                num.erase(0, num.find_first_not_of(" \t"));
                num.erase(num.find_last_not_of(" \t") + 1);
                if (!num.empty()) {
                    row.push_back(stod(num));
                }
            }
            
            if (!row.empty()) {
                if (in_token_embeddings && row_idx < token_embedding_table.size()) {
                    token_embedding_table[row_idx] = row;
                    row_idx++;
                } else if (in_position_embeddings && row_idx < position_embedding_table.size()) {
                    position_embedding_table[row_idx] = row;
                    row_idx++;
                }
            }
        }
    }
    
    file.close();
    cout << "Model loaded from " << filename << endl;
    return true;
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
    for (size_t i = 0; i < logits.size(); ++i)
    {
        double max_val = *max_element(logits[i].begin(), logits[i].end());
        double sum = 0.0;
        for (size_t j = 0; j < logits[i].size(); ++j)
        {
            probs[i][j] = exp(logits[i][j] - max_val);
            sum += probs[i][j];
        }
        for (size_t j = 0; j < logits[i].size(); ++j)
        {
            probs[i][j] /= sum;
        }
    }
    return probs; // FIXED: return probs instead of logits
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