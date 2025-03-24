#include "attentionmechanism.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

Linear::Linear(int in_features, int out_features)
{
    weights.resize(out_features, vector<double>(in_features));
    biases.resize(out_features, 0.0);
    initialize_weights();
}

vector<double> Linear::forward(const vector<double> &x)
{
    vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        for (size_t j = 0; j < weights[0].size(); ++j)
        {
            output[i] += weights[i][j] * x[j];
        }
        output[i] += biases[i];
    }
    return output;
}

vector<vector<vector<double>>> Linear::forward(const vector<vector<vector<double>>> &x)
{
    vector<vector<vector<double>>> output(x.size(), vector<vector<double>>(x[0].size(), vector<double>(weights.size(), 0.0)));
    for (size_t i = 0; i < x.size(); ++i)
    {
        for (size_t j = 0; j < x[0].size(); ++j)
        {
            output[i][j] = forward(x[i][j]);
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

Head::Head(int head_size) : key(head_size, head_size), query(head_size, head_size), value(head_size, head_size), dropout(0.2) {}

vector<double> Head::forward(const vector<double> &x)
{
    vector<double> k = key.forward(x);
    vector<double> q = query.forward(x);
    vector<double> v = value.forward(x);

    double scale = 1.0 / sqrt(k.size());
    vector<double> scores(k.size(), 0.0);
    for (size_t i = 0; i < k.size(); ++i)
    {
        scores[i] = q[i] * k[i] * scale;
    }

    double max_score = *max_element(scores.begin(), scores.end());
    for (auto &score : scores)
    {
        score = exp(score - max_score);
    }
    double sum_scores = accumulate(scores.begin(), scores.end(), 0.0);
    for (auto &score : scores)
    {
        score /= sum_scores;
    }

    vector<double> weighted_sum(v.size(), 0.0);
    for (size_t i = 0; i < v.size(); ++i)
    {
        weighted_sum[i] = scores[i] * v[i];
    }

    return dropout.forward(weighted_sum);
}

MultiHeadAttention::MultiHeadAttention(int n_head, int head_size) : output_linear(n_head * head_size, n_head * head_size)
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
    return output_linear.forward(concat_heads);
}

LayerNorm::LayerNorm(int n_embd) : n_embd(n_embd)
{
    gamma.resize(n_embd, 1.0);
    beta.resize(n_embd, 0.0);
}

vector<double> LayerNorm::forward(const vector<double> &x)
{

    double mean = accumulate(x.begin(), x.end(), 0.0) / x.size();
    double variance = 0.0;
    for (const auto &val : x)
    {
        variance += (val - mean) * (val - mean);
    }
    variance /= x.size();
    double stddev = sqrt(variance + 1e-5);

    vector<double> normalized(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        normalized[i] = (x[i] - mean) / stddev;
    }

    vector<double> output(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        output[i] = gamma[i] * normalized[i] + beta[i];
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

FeedForward::FeedForward(int n_embd) : linear1(n_embd, 4 * n_embd), linear2(4 * n_embd, n_embd) {}

vector<double> FeedForward::forward(const vector<double> &x)
{
    vector<double> hidden = linear1.forward(x);
    for (auto &val : hidden)
    {
        val = max(0.0, val); // ReLU activation
    }
    return linear2.forward(hidden);
}

Block::Block(int n_embd, int n_head) : sa(n_head, n_embd / n_head), ffwd(n_embd), ln1(n_embd), ln2(n_embd) {}

vector<double> Block::forward(const vector<double> &x)
{
    vector<double> x1 = ln1.forward(x);
    vector<double> sa_output = sa.forward(x1);
    vector<double> x2 = ln2.forward(x1);
    vector<double> ffwd_output = ffwd.forward(x2);
    vector<double> output(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        output[i] = x[i] + sa_output[i] + ffwd_output[i];
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