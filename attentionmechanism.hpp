#ifndef ATTENTIONMECHANISM_HPP
#define ATTENTIONMECHANISM_HPP

#include <vector>
#include <random>

using namespace std;

class Linear
{
public:
    Linear(int in_features, int out_features);
    vector<double> forward(const vector<double> &x);
    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>> &x);

private:
    void initialize_weights();
    vector<vector<double>> weights;
    vector<double> biases;
};

class Dropout
{
public:
    Dropout(double p);
    vector<double> forward(const vector<double> &x);

private:
    double p;
};

class Head
{
public:
    Head(int head_size);
    vector<double> forward(const vector<double> &x);

private:
    Linear key;
    Linear query;
    Linear value;
    Dropout dropout;
};

class MultiHeadAttention
{
public:
    MultiHeadAttention(int n_head, int head_size);
    vector<double> forward(const vector<double> &x);

private:
    vector<Head> heads;
    Linear output_linear;
};

class LayerNorm
{
public:
    LayerNorm(int n_embd);
    vector<double> forward(const vector<double> &x);
    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>> &x);

private:
    int n_embd;
    vector<double> gamma;
    vector<double> beta;
};

class FeedForward
{
public:
    FeedForward(int n_embd);
    vector<double> forward(const vector<double> &x);

private:
    Linear linear1;
    Linear linear2;
};

class Block
{
public:
    Block(int n_embd, int n_head);
    vector<double> forward(const vector<double> &x);
    vector<vector<double>> forward(const vector<vector<double>> &x);
    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>> &x);

private:
    MultiHeadAttention sa;
    FeedForward ffwd;
    LayerNorm ln1;
    LayerNorm ln2;
};

#endif // ATTENTIONMECHANISM_HPP