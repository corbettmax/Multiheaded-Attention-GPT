#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <iterator>
#include <fstream>
#include "./util.cpp"
#include <chrono>

using namespace std;

// Softmax function to calculate attention weights
// Reminder to self: softmax is a function that takes a vector of values and returns a vector of values that sum to 1
vector<double> softmax(const vector<double>& x) {
    vector<double> exp_x(x.size());
    double max_x = *max_element(x.begin(), x.end());

    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = exp(x[i] - max_x);
    }

    double sum_exp_x = accumulate(exp_x.begin(), exp_x.end(), 0.0);

    for (size_t i = 0; i < exp_x.size(); ++i) {
        exp_x[i] /= sum_exp_x;
    }

    return exp_x;
}

// Encode words into vectors
void createWordRepresentations(const vector<string>& sentences, unordered_map<string, int>& wordToIndex, unordered_map<int, string>& indexToWord, vector<vector<double>>& wordEmbeddings) {
    for (const auto& sentence : sentences) {
        istringstream iss(sentence);
        string word;
        while (iss >> word) {
            if (wordToIndex.find(word) == wordToIndex.end()) {
                int index = wordToIndex.size();
                wordToIndex[word] = index;
                indexToWord[index] = word;
                vector<double> embedding(3);
                generate(embedding.begin(), embedding.end(), []() { return static_cast<double>(rand()) / RAND_MAX; });
                wordEmbeddings.push_back(embedding);
            }
        }
    }
}

// Calculate the self-attention of a node
vector<double> calculateSelfAttention(const vector<double>& query, const vector<vector<double>>& keys, const vector<vector<double>>& values) {
    vector<double> scores(keys.size());

    // Take the dot product of the query with each key
    for (size_t i = 0; i < keys.size(); ++i) {
        scores[i] = inner_product(query.begin(), query.end(), keys[i].begin(), 0.0) / sqrt(keys[i].size());
    }

    vector<double> attentionWeights = softmax(scores);
    return attentionWeights;
}

// Predict next word with self-attention
pair<string, vector<double>> predictNextWordWithSelfAttention(const string& currentWord, const vector<string>& contextWindow, const vector<string>& words, const vector<vector<double>>& wordEmbeddings, const unordered_map<string, int>& wordToIndex, const unordered_map<int, string>& indexToWord) {
    
    // Calculate the context embedding
    vector<vector<double>> contextEmbeddings;
    for (const auto& word : contextWindow) {
if (wordToIndex.find(word) != wordToIndex.end()) {
        contextEmbeddings.push_back(wordEmbeddings[wordToIndex.at(word)]);
}
    }

    if (contextEmbeddings.empty()) {
        cerr << "Error: Context embeddings are empty." << endl;
        return {"", {}};
    }

    // Calculate the query by using the same context embedding
    vector<double> query(contextEmbeddings[0].size(), 0.0);
    for (const auto& embedding : contextEmbeddings) {
        transform(query.begin(), query.end(), embedding.begin(), query.begin(), plus<double>());
    }
    for (auto& val : query) {
        val /= contextEmbeddings.size();
    }

    vector<vector<double>> keys, values;
    for (const auto& word : words) {
    if (wordToIndex.find(word) != wordToIndex.end()) {
        keys.push_back(wordEmbeddings[wordToIndex.at(word)]);
        values.push_back(wordEmbeddings[wordToIndex.at(word)]);
    }
    }

    if (keys.empty() || values.empty()) {
        cerr << "Error: Keys or values are empty." << endl;
        return {"", {}};
    }

    vector<double> attentionWeights = calculateSelfAttention(query, keys, values);
    int predictedIndex = distance(attentionWeights.begin(), max_element(attentionWeights.begin(), attentionWeights.end()));
    string predictedWord = indexToWord.at(predictedIndex);

    return {predictedWord, attentionWeights};
}

void saveQueriesKeysValues(const vector<double>& query, const vector<vector<double>>& keys, const vector<vector<double>>& values, const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    outFile << "Query:\n";
    for (const auto& q : query) {
        outFile << q << " ";
    }
    outFile << "\n\nKeys:\n";
    for (const auto& key : keys) {
        for (const auto& k : key) {
            outFile << k << " ";
        }
        outFile << "\n";
    }
    outFile << "\nValues:\n";
    for (const auto& value : values) {
        for (const auto& v : value) {
            outFile << v << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

int main() {
    srand(time(0));

    auto start = chrono::high_resolution_clock::now();

    string inputFilename = "./input.txt";
    string outputFilename = "./output.txt";
    if (!fileExists(inputFilename)) {
        cerr << "Error: Input file does not exist." << endl;
        return 1;
    }
    if (!fileExists(outputFilename)) {
        cout << "Tokenizing text..." << endl;
        cleanText(inputFilename, outputFilename);
        
    }
    vector<string> sentences = loadSentences(outputFilename);

    unordered_map<string, int> wordToIndex;
    unordered_map<int, string> indexToWord;
    vector<vector<double>> wordEmbeddings;

    createWordRepresentations(sentences, wordToIndex, indexToWord, wordEmbeddings);

    string currentWord = "o";
    int contextWindowSize = 3;  // Considering two words before the current word

    for (const auto& sentence : sentences) {
        if (sentence.find(currentWord) == string::npos) {
            continue;
        }
        istringstream iss(sentence);
        vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
        auto it = find(words.begin(), words.end(), currentWord);
        if (it == words.end()) {
            continue;
        }
        int currentWordIndex = distance(words.begin(), it);
        vector<string> contextWindow(words.begin() + max(0, currentWordIndex - contextWindowSize), words.begin() + currentWordIndex);

        auto [predictedWord, attentionProbabilities] = predictNextWordWithSelfAttention(currentWord, contextWindow, words, wordEmbeddings, wordToIndex, indexToWord);

        if (predictedWord.empty()) {
            continue;
        }

        cout << "\nGiven the word: " << currentWord << endl;
        cout << "Context: ";
        for (const auto& word : contextWindow) {
            cout << word << " ";
        }
        cout << endl;
        cout << "Sentence: " << sentence << endl;
        cout << "Attention Probabilities:" << endl;
        for (size_t i = 0; i < words.size(); ++i) {
            cout << "\t" << words[i] << ": " << attentionProbabilities[i] << endl;
        }
        cout << "Predicted next word: " << predictedWord << endl;

        // Save queries, keys, and values
        vector<double> query(contextWindow.size(), 0.0);
        for (const auto& embedding : contextWindow) {
            transform(query.begin(), query.end(), embedding.begin(), query.begin(), plus<double>());
        }
        for (auto& val : query) {
            val /= contextWindow.size();
        }
        vector<vector<double>> keys, values;
        for (const auto& word : words) {
            if (wordToIndex.find(word) != wordToIndex.end()) {
                keys.push_back(wordEmbeddings[wordToIndex.at(word)]);
                values.push_back(wordEmbeddings[wordToIndex.at(word)]);
            }
        }
        saveQueriesKeysValues(query, keys, values, "queries_keys_values.txt");
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}