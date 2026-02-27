#include "attentionmechanism.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>

using namespace std;

Linear::Linear(int in_features, int out_features)
{
    weights.resize(out_features, vector<double>(in_features));
    biases.resize(out_features, 0.0);
    initialize_weights();
}

vector<double> Linear::forward(const vector<double> &x)
{
    int out_features = weights.size();
    int in_features = weights[0].size();
    vector<double> output(out_features, 0.0);
    
    // GPU-accelerated matrix-vector multiplication
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_features; ++i)
    {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < in_features; ++j)
        {
            sum += weights[i][j] * x[j];
        }
        output[i] = sum + biases[i];
    }
    return output;
}

vector<vector<vector<double>>> Linear::forward(const vector<vector<vector<double>>> &x)
{
    int B = x.size();
    int T = x[0].size();
    int in_features = weights[0].size();
    int out_features = weights.size();
    
    vector<vector<vector<double>>> output(B, vector<vector<double>>(T, vector<double>(out_features, 0.0)));
    
    // Parallelize over batch and time dimensions
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b)
    {
        for (int t = 0; t < T; ++t)
        {
            for (int i = 0; i < out_features; ++i)
            {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int j = 0; j < in_features; ++j)
                {
                    sum += weights[i][j] * x[b][t][j];
                }
                output[b][t][i] = sum + biases[i];
            }
        }
    }
    return output;
}

void Linear::initialize_weights()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, 0.02);
    for (auto &row : weights)
    {
        generate(row.begin(), row.end(), [&]()
                 { return d(gen); });
    }
}

Dropout::Dropout(double p) : p(p) {}

vector<double> Dropout::forward(const vector<double> &x)
{
    vector<double> output = x;
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution d(1.0 - p);
    for (auto &val : output)
    {
        val *= d(gen);
        val /= 1.0 - p;
    }
    
    return output;
}

Head::Head(int head_size) : key(head_size, head_size), query(head_size, head_size), value(head_size, head_size), dropout(0.2), head_size(head_size) {}

vector<double> Head::forward(const vector<double> &x)
{
    vector<double> k = key.forward(x);
    vector<double> q = query.forward(x);
    vector<double> v = value.forward(x);

    // For single vector, just compute scaled dot product with itself
    double scale = 1.0 / sqrt(head_size);
    double score = 0.0;
    for (size_t i = 0; i < k.size(); ++i)
    {
        score += q[i] * k[i];
    }
    score *= scale;
    
    double attention_weight = 1.0; // Single token attends to itself with weight 1

    vector<double> weighted_sum(v.size(), 0.0);
    for (size_t i = 0; i < v.size(); ++i)
    {
        weighted_sum[i] = attention_weight * v[i];
    }

    return dropout.forward(weighted_sum);
}

vector<vector<vector<double>>> Head::forward(const vector<vector<vector<double>>> &x)
{
    // x shape: (B, T, C)
    int B = x.size();
    int T = x[0].size();
    int C = head_size;
    
    // Compute Q, K, V for all positions
    vector<vector<vector<double>>> Q(B, vector<vector<double>>(T, vector<double>(C)));
    vector<vector<vector<double>>> K(B, vector<vector<double>>(T, vector<double>(C)));
    vector<vector<vector<double>>> V(B, vector<vector<double>>(T, vector<double>(C)));
    
    // Parallelize Q, K, V computation
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            Q[b][t] = query.forward(x[b][t]);
            K[b][t] = key.forward(x[b][t]);
            V[b][t] = value.forward(x[b][t]);
        }
    }
    
    // Compute attention scores: Q @ K^T
    double scale = 1.0 / sqrt(C);
    vector<vector<vector<double>>> output(B, vector<vector<double>>(T, vector<double>(C, 0.0)));
    
    // Parallelize across batches
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        // Compute attention weights for this batch
        vector<vector<double>> attn_weights(T, vector<double>(T, 0.0));
        
        // Compute all attention scores
        for (int i = 0; i < T; ++i) {
            double max_score = -1e9;
            
            // Compute Q[i] @ K[j]^T with SIMD
            for (int j = 0; j <= i; ++j) { // Causal mask
                double score = 0.0;
                #pragma omp simd reduction(+:score)
                for (int k = 0; k < C; ++k) {
                    score += Q[b][i][k] * K[b][j][k];
                }
                score *= scale;
                attn_weights[i][j] = score;
                max_score = max(max_score, score);
            }
            
            // Apply softmax (with numerical stability)
            double sum_exp = 0.0;
            for (int j = 0; j <= i; ++j) {
                attn_weights[i][j] = exp(attn_weights[i][j] - max_score);
                sum_exp += attn_weights[i][j];
            }
            double inv_sum = 1.0 / sum_exp;
            for (int j = 0; j <= i; ++j) {
                attn_weights[i][j] *= inv_sum;
            }
        }
        
        // Compute weighted sum of values (attention_weights @ V)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j <= i; ++j) {
                double weight = attn_weights[i][j];
                #pragma omp simd
                for (int k = 0; k < C; ++k) {
                    output[b][i][k] += weight * V[b][j][k];
                }
            }
        }
    }
    
    // Apply dropout to output
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            output[b][t] = dropout.forward(output[b][t]);
        }
    }
    
    return output;
}

MultiHeadAttention::MultiHeadAttention(int n_head, int head_size) : output_linear(n_head * head_size, n_head * head_size), dropout(0.2)
{
    for (int i = 0; i < n_head; ++i)
    {
        heads.push_back(Head(head_size));
    }
}

vector<double> MultiHeadAttention::forward(const vector<double> &x)
{
    vector<double> concat_heads;
    for (auto &head : heads)
    {
        vector<double> head_output = head.forward(x);
        concat_heads.insert(concat_heads.end(), head_output.begin(), head_output.end());
    }
    return dropout.forward(output_linear.forward(concat_heads));
}

vector<vector<vector<double>>> MultiHeadAttention::forward(const vector<vector<vector<double>>> &x)
{
    // x shape: (B, T, C)
    int B = x.size();
    int T = x[0].size();
    
    // Run each head
    vector<vector<vector<vector<double>>>> head_outputs;
    for (auto &head : heads) {
        head_outputs.push_back(head.forward(x));
    }
    
    // Concatenate heads along channel dimension
    int head_dim = head_outputs[0][0][0].size();
    int total_dim = head_outputs.size() * head_dim;
    
    vector<vector<vector<double>>> concat(B, vector<vector<double>>(T, vector<double>(total_dim)));
    
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int offset = 0;
            for (size_t h = 0; h < head_outputs.size(); ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    concat[b][t][offset + d] = head_outputs[h][b][t][d];
                }
                offset += head_dim;
            }
        }
    }
    
    // Apply output projection
    vector<vector<vector<double>>> output = output_linear.forward(concat);
    
    // Apply dropout
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            output[b][t] = dropout.forward(output[b][t]);
        }
    }
    
    return output;
}

LayerNorm::LayerNorm(int n_embd) : n_embd(n_embd)
{
    gamma.resize(n_embd, 1.0);
    beta.resize(n_embd, 0.0);
}

vector<double> LayerNorm::forward(const vector<double> &x)
{
    int n = x.size();
    
    // Compute mean
    double mean = 0.0;
    #pragma omp simd reduction(+:mean)
    for (int i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= n;
    
    // Compute variance
    double variance = 0.0;
    #pragma omp simd reduction(+:variance)
    for (int i = 0; i < n; ++i) {
        double diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    double inv_stddev = 1.0 / sqrt(variance + 1e-5);

    // Normalize and scale
    vector<double> output(n);
    #pragma omp simd
    for (int i = 0; i < n; ++i) {
        output[i] = gamma[i] * (x[i] - mean) * inv_stddev + beta[i];
    }
    return output;
}

vector<vector<vector<double>>> LayerNorm::forward(const vector<vector<vector<double>>> &x)
{
    vector<vector<vector<double>>> output = x;
    for (auto &batch : output)
    {
        for (auto &seq : batch)
        {
            seq = forward(seq);
        }
    }
    return output;
}

FeedForward::FeedForward(int n_embd) : linear1(n_embd, 4 * n_embd), linear2(4 * n_embd, n_embd), dropout(0.2) {}

vector<double> FeedForward::forward(const vector<double> &x)
{
    vector<double> hidden = linear1.forward(x);
    for (auto &val : hidden)
    {
        val = max(0.0, val); // ReLU activation
    }
    return dropout.forward(linear2.forward(hidden));
}

vector<vector<vector<double>>> FeedForward::forward(const vector<vector<vector<double>>> &x)
{
    int B = x.size();
    int T = x[0].size();
    
    vector<vector<vector<double>>> hidden = linear1.forward(x);
    
    // Apply ReLU in parallel
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (size_t i = 0; i < hidden[b][t].size(); ++i) {
                hidden[b][t][i] = max(0.0, hidden[b][t][i]);
            }
        }
    }
    
    vector<vector<vector<double>>> output = linear2.forward(hidden);
    
    // Apply dropout in parallel
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            output[b][t] = dropout.forward(output[b][t]);
        }
    }
    
    return output;
}

Block::Block(int n_embd, int n_head) : sa(n_head, n_embd / n_head), ffwd(n_embd), ln1(n_embd), ln2(n_embd) {}

vector<double> Block::forward(const vector<double> &x)
{
    vector<double> x1 = ln1.forward(x);
    vector<double> sa_output = sa.forward(x1);
    
    // Residual connection 1
    vector<double> x2(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x2[i] = x[i] + sa_output[i];
    }
    
    vector<double> x3 = ln2.forward(x2);
    vector<double> ffwd_output = ffwd.forward(x3);
    
    // Residual connection 2
    vector<double> output(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = x2[i] + ffwd_output[i];
    }
    return output;
}

vector<vector<double>> Block::forward(const vector<vector<double>> &x)
{
    vector<vector<double>> output = x;
    for (auto &batch : output)
    {
        batch = forward(batch);
    }
    return output;
}

vector<vector<vector<double>>> Block::forward(const vector<vector<vector<double>>> &x)
{
    // Apply layer norm, attention, and residual
    vector<vector<vector<double>>> x1 = ln1.forward(x);
    vector<vector<vector<double>>> sa_output = sa.forward(x1);
    
    // Residual connection 1
    int B = x.size();
    int T = x[0].size();
    int C = x[0][0].size();
    
    vector<vector<vector<double>>> x2(B, vector<vector<double>>(T, vector<double>(C)));
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                x2[b][t][c] = x[b][t][c] + sa_output[b][t][c];
            }
        }
    }
    
    // Apply layer norm and feedforward
    vector<vector<vector<double>>> x3 = ln2.forward(x2);
    vector<vector<vector<double>>> ffwd_output = ffwd.forward(x3);
    
    // Residual connection 2
    vector<vector<vector<double>>> output(B, vector<vector<double>>(T, vector<double>(C)));
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                output[b][t][c] = x2[b][t][c] + ffwd_output[b][t][c];
            }
        }
    }
    
    return output;
}