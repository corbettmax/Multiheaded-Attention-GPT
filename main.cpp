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

    string inputFilename = "./input.txt";
    string outputFilename = "./output.txt";
    if (!fileExists(inputFilename))
    {
        //cerr << "Error: Input file does not exist." << endl;
        //return 1;
    }
    if (!fileExists(outputFilename))
    {
        //cout << "Tokenizing text..." << endl;
        //cleanText(inputFilename, outputFilename);
    }

    loadSentences(inputFilename);
    splitDataset(encoded_data, 0.5); // 10% training, 90% testing
    GPTLanguageModel gpt(vocab_size, n_embd, block_size, n_layer, n_head);

    // Training Loop
    // Activate once backward function is implemented
    /*for (int iter = 0; iter < max_iters; ++iter)
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

        // Evaluate the loss
        auto logits_loss = gpt.forward(x, &y);
        auto &logits = std::get<0>(logits_loss);
        auto &loss = std::get<1>(logits_loss);
        //gpt.backward(loss);
    }*/

    //cout << "Finished training over " << max_iters << " iterations" << endl;
    //auto training_end = chrono::high_resolution_clock::now();
    //chrono::duration<double> training_elapsed = training_end - start;
    //cout << "Training elapsed time: " << training_elapsed.count() << " seconds" << endl;


    // Generate from the model
    vector<vector<int>> context = {{5, 6, 7, 8, 9}, {3894, 3895, 96, 300, 3898}};
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