#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm> 
#include "multiheadedgpt.hpp"

using namespace std;

extern int batch_size; // how many independent sequences will we process in parallel?
extern int block_size; // what is the maximum context length for predictions?
extern int max_iters;
extern int eval_interval;
extern double learning_rate;
extern string device;
extern int eval_iters;
extern int n_embd;
extern int n_head;
extern int n_layer;
extern int dropout;

extern int vocab_size;
extern unordered_map<string, int> wordtoindex;
extern unordered_map<int, string> indextoword;
extern vector<vector<int>> encoded_data;
extern vector<vector<int>> train_data;
extern vector<vector<int>> val_data;

vector<int> encode(const string &s);
string decode(const vector<int> &l);
void loadSentences(const string &filename);
vector<string> cleanText(const string &inputFilename, const string &outputFilename);
bool fileExists(const string &filename);
vector<vector<double>> mulMat(const vector<vector<double>> &mat1, const vector<vector<double>> &mat2);
vector<vector<double>> transpose(const vector<vector<double>> &mat);
void getBatch(const string &split, vector<vector<int>> &x, vector<vector<int>> &y);
unordered_map<string, double> estimateLoss(GPTLanguageModel &model);
void splitDataset(const vector<vector<int>>& dataset, double trainRatio);

#endif // UTIL_HPP