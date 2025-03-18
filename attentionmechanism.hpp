#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

class AttentionMechanism {
public:
    AttentionMechanism();
    std::vector<double> softmax(const std::vector<double>& x);
    void createWordRepresentations(const std::vector<std::string>& sentences);
    std::vector<double> calculateSelfAttention(const std::vector<double>& query, const std::vector<std::vector<double>>& keys, const std::vector<std::vector<double>>& values);
    std::pair<std::string, std::vector<double>> predictNextWordWithSelfAttention(const std::string& currentWord, const std::vector<std::string>& contextWindow, const std::vector<std::string>& words);
    void saveQueriesKeysValues(const std::vector<double>& query, const std::vector<std::vector<double>>& keys, const std::vector<std::vector<double>>& values, const std::string& filename);

private:
    std::unordered_map<std::string, int> wordToIndex;
    std::unordered_map<int, std::string> indexToWord;
    std::vector<std::vector<double>> wordEmbeddings;
};