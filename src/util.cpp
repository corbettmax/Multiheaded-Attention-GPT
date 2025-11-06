#include "./util.hpp"

using namespace std;

int batch_size = 16; // how many independent sequences will we process in parallel?
int block_size = 64; // what is the maximum context length for predictions?
int max_iters = 1;
int eval_interval = 10;
double learning_rate = 0.01;
string device = "cpu";
int eval_iters = 10;
int n_embd = 384;
int n_head = 6;
int n_layer = 6;
int dropout = 0.2;

int vocab_size;
unordered_map<string, int> wordtoindex;
unordered_map<int, string> indextoword;
vector<vector<int>> encoded_data;
vector<vector<int>> train_data; // Placeholder for training data
vector<vector<int>> val_data; // Placeholder for validation data
vector<vector<int>> wordEmbeddings; // Placeholder for word embeddings

// Function to decode a list of integers to a string
string decode(const vector<int> &l) {
    string decoded;
    for (int i : l) {
        decoded += indextoword[i] + " ";
    }
    return decoded;
}

// Function to encode a string to a list of ints
vector<int> encode(const string &s) {
    vector<int> encoded;
    istringstream iss(s);
    string word;
    while (iss >> word) {
        encoded.push_back(wordtoindex[word]);
    }
    return encoded;
}

// Function to load data from a file
void loadSentences(const string &filename) {
    vector<string> sentences;
    ifstream file(filename);
    string line, lowercaseline;

    while (getline(file, line)) {
        line.erase(line.find_last_not_of(" \n\r\t")+1);
        sentences.push_back(line);
    }

    for (const auto& sentence : sentences) {
        istringstream iss(sentence);
        string word;
        while (iss >> word) {
            if (wordtoindex.find(word) == wordtoindex.end()) {
                int index = wordtoindex.size();
                wordtoindex[word] = index;
                indextoword[index] = word;
                encoded_data.push_back(encode(word));
            }
        }
    }
    vocab_size = wordtoindex.size();

}

vector<string> cleanText(const string& inputFilename, const string& outputFilename) {
    vector<string> sentences;
    ifstream inputFile(inputFilename);
    ofstream outputFile(outputFilename);
    string line, lowercaseline;

    while (getline(inputFile, line)) {
        // Trim leading and trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        // Check if the line is not blank and does not contain a colon
        if (!line.empty() && line.find(':') == string::npos) {
            lowercaseline.clear();
            for (auto x : line) {
                if (ispunct(x)) continue;
                lowercaseline += tolower(x);
            }
            sentences.push_back(lowercaseline);
            outputFile << lowercaseline << endl;
        }
    }

    return sentences;
}

// Function to check if a file exists
bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

vector<vector<double>> mulMat(const vector<vector<double>> &mat1,const vector<vector<double>> &mat2)
{   
    vector<vector<double>> rslt(mat1.size(), vector<double>(mat2[0].size(), 0));
    
    for (int i = 0; i < mat1.size(); i++)
    {
        for (int j = 0; j < mat2[0].size(); j++)
        {
            for (int k = 0; k < mat2.size(); k++)
            {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return rslt;
}

vector<vector<double>> transpose(const vector<vector<double>> &mat)
{
    vector<vector<double>> rslt(mat[0].size(), vector<double>(mat.size(), 0));
    
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            rslt[j][i] = mat[i][j];
        }
    }
    return rslt;
}

// Function to generate a small batch of data of inputs x and targets y
void getBatch(const string &split, vector<vector<int>> &x, vector<vector<int>> &y) {
    const vector<vector<int>>& data = (split == "train") ? train_data : val_data;

    cout << "Data size: " << data.size() << " x " << data[0].size() << endl;

    for (int i = 0; i < batch_size; ++i) {
        int start_idx = rand() % (data.size() - block_size); 
        for (int j = 0; j < block_size; ++j) {
            x[i][j] = data[start_idx][j];
            y[i][j] = data[start_idx][j + 1];
        }
    }

    cout << "Batch generated successfully" << endl;
    cout << "Batch size: " << x.size() << " x " << x[0].size() << endl;
    return;
}

// Function to estimate loss
unordered_map<string, double> estimateLoss(GPTLanguageModel &model) {
    unordered_map<string, double> out;
    int eval_iters = 10; // Example evaluation iterations
    for (const string &split : {"train", "val"}) {
        double total_loss = 0.0;
        for (int k = 0; k < eval_iters; ++k) {
            vector<vector<int>> X(batch_size, vector<int>(block_size));
            vector<vector<int>> Y(batch_size, vector<int>(block_size));
            getBatch(split, X, Y);

            // Assuming model.forward returns a pair of logits and loss
            auto [logits, loss] = model.forward(X, &Y);
            total_loss += loss;
        }
        out[split] = total_loss / eval_iters;
    }
    return out;
}


void splitDataset(const vector<vector<int>>& dataset, double trainRatio) {
    size_t trainSize = static_cast<size_t>(dataset.size() * trainRatio);
    train_data.assign(dataset.begin(), dataset.begin() + trainSize);
    val_data.assign(dataset.begin() + trainSize, dataset.end());
}