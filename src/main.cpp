#include <chrono>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include "./multiheadedgpt.hpp"
#include "./util.hpp"

using namespace std;

int main()
{

    auto start = chrono::high_resolution_clock::now();

    string inputFilename = "training/input.txt";
    string outputFilename = "output.txt";
    if (!fileExists(inputFilename))
    {
        cerr << "Error: Input file does not exist." << endl;
        return 1;
    }
    if (!fileExists(outputFilename))
    {
        cout << "Tokenizing text..." << endl;
        cleanText(inputFilename, outputFilename);
    }

    loadSentences(inputFilename);
    splitDataset(encoded_data, 0.5); // 50% training, 50% validation
    GPTLanguageModel gpt(vocab_size, n_embd, block_size, n_layer, n_head);

    // Try to load existing model
    string model_file = "model_weights.json";
    bool skip_training = false;
    if (fileExists(model_file)) {
        cout << "Found existing model file. Load it? (y/n): ";
        char choice;
        cin >> choice;
        if (choice == 'y' || choice == 'Y') {
            if (gpt.load_model(model_file)) {
                skip_training = true;
                cout << "Model loaded successfully. Skipping training." << endl;
            }
        }
    }

    // Training Loop
    if (!skip_training) {
        cout << "\nStarting training..." << endl;
    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval == 0 || iter == max_iters - 1)
        {
            auto losses = estimateLoss(gpt);
            cout << "step " << iter << ": train loss " << losses["train"] << ", val loss " << losses["val"] << endl;
        }

        vector<vector<int>> x(batch_size, vector<int>(block_size));
        vector<vector<int>> y(batch_size, vector<int>(block_size));
        // Sample a batch of data
        getBatch("train", x, y);

        // Train: forward + backward pass
        gpt.backward(x, y, learning_rate);
    }

    cout << "Finished training over " << max_iters << " iterations" << endl;
    auto training_end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_elapsed = training_end - start;
    cout << "Training elapsed time: " << training_elapsed.count() << " seconds" << endl;

    // Save trained model
    gpt.save_model("model_weights.json");
    }

    // Generate from the model
    vector<vector<int>> context = {{5, 6, 7, 8, 9}};
    vector<vector<int>> idx = gpt.generate(context, 10);
    cout << "Generated text:" << endl;
    for (auto &seq : idx)
    {
        cout << decode(seq) << " " << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}