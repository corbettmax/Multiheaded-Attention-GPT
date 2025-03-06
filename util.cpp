#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>

using namespace std;


// Load sentences from a formatted txt file
vector<string> loadSentences(const string& filename) {
    vector<string> sentences;
    ifstream file(filename);
    string line, lowercaseline;

    while (getline(file, line)) {
        line.erase(line.find_last_not_of(" \n\r\t")+1);
        sentences.push_back(line);
    }

    return sentences;
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