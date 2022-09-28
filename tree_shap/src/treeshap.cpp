#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip> 

using namespace std;

template <typename T>
using Matrix = vector<vector<T>>;
template <typename T>
using Tensor = vector<vector<vector<T>>>;


template<typename T>
Matrix<T> readFile(string filename){
    ifstream ifs(filename);
    vector<vector<T>> data;
    int N(0);
    for (string line; getline(ifs, line); ) {
        stringstream ss(line);
        data.push_back(vector<T> ());
        double feature;
        while (ss >> feature){
            data[N].push_back(feature);
        }
        N++;
    }
    return data;
}


void compute_W(Matrix<double> &W)
{
    int D = W.size();
    for (double j(0); j < D; j++){
        W[0][j] = 1 / (j + 1);
        W[j][j] = 1 / (j + 1);
    }
    for (double j(2); j < D; j++){
        for (double i(j-1); i > 0; i--){
            W[i][j] = (j - i) / (i + 1) * W[i+1][j];
        }
    }
}


// Recursion function
pair<double, double> recurse(int n,
                            vector<double> &x_f, vector<double> &x_b,
                            Matrix<int> &categorical_to_features,
                            vector<int> &features,
                            vector<int> &child_left,
                            vector<int> &child_right,
                            vector<double> &threshold,
                            vector<double> &value,
                            vector<vector<double>> &W,
                            int &n_features,
                            vector<double> &phi,
                            vector<int> &in_SAB,
                            vector<int> &in_SA)
{
    int current_feature = features[n];
    int xf_child(0), xb_child(0);
    // Case 1: Leaf
    if (child_left[n] < 0)
    {
        double pos(0.0), neg(0.0);
        // S_AB is empty
        if (in_SAB[n_features] == 0)
        {
            return make_pair(pos, neg);
        }
        // SA is non-empty
        if (in_SA[n_features] > 0)
        {
            pos = W[in_SA[n_features]-1][in_SAB[n_features]-1] * value[n];
        }
        // SB is non-empty
        if (in_SA[n_features] < in_SAB[n_features])
        {
            neg = W[in_SA[n_features]][in_SAB[n_features]-1] * value[n];
        }
        return make_pair(pos, neg);
    }

    // Find children associated with xf and xb
    if (x_f[current_feature] <= threshold[n]){
        xf_child = child_left[n];
    } else {xf_child = child_right[n];}
    if (x_b[current_feature] <= threshold[n]){
        xb_child = child_left[n];
    } else {xb_child = child_right[n];}

    // Case 2: xf and xb go the same way
    if (xf_child == xb_child){
        return recurse(xf_child, x_f, x_b, categorical_to_features, features, 
                       child_left, child_right, threshold, value, W, n_features, phi, in_SAB, in_SA);
    }
    // Case 3: Feature encountered before in SAB
    if (in_SAB[ categorical_to_features[current_feature][0] ] > 0){
        if (in_SA[ categorical_to_features[current_feature][0] ] > 0){
            return recurse(xf_child, x_f, x_b, categorical_to_features, features, 
                           child_left, child_right, threshold, value, W, n_features, phi, in_SAB, in_SA);
        }
        else{
            return recurse(xb_child, x_f, x_b, categorical_to_features, features, 
                           child_left, child_right, threshold, value, W, n_features, phi, in_SAB, in_SA);
        }
    }

    // Case 4: xf and xb don't go the same way
    else {
        // Go down xf_child
        in_SA[ categorical_to_features[current_feature][0] ]++; in_SA[n_features]++;
        in_SAB[ categorical_to_features[current_feature][0] ]++; in_SAB[n_features]++;
        pair<double, double> pairf = recurse(xf_child, x_f, x_b, categorical_to_features, features, 
                                             child_left, child_right, threshold, value, W, n_features, phi, in_SAB, in_SA);

        // Go down xb_child
        in_SA[ categorical_to_features[current_feature][0] ]--; in_SA[n_features]--;
        pair<double, double> pairb = recurse(xb_child, x_f, x_b, categorical_to_features, features, 
                                             child_left, child_right, threshold, value, W, n_features, phi, in_SAB, in_SA);
        in_SAB[ categorical_to_features[current_feature][0] ]--; in_SAB[n_features]--;

        // Add contribution to the feature
        phi[ categorical_to_features[current_feature][0] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}


Tensor<double> treeSHAP(Matrix<double> &X_f, 
                        Matrix<double> &X_b, 
                        Matrix<int> categorical_to_features,
                        Matrix<int> &features,
                        Matrix<int> &child_left,
                        Matrix<int> &child_right,
                        Matrix<double> &threshold,
                        Matrix<double> &value,
                        Matrix<double> &W)
{
    // Setup
    int n_features = categorical_to_features[categorical_to_features.size()-1][0] + 1;
    int n_trees = features.size();
    int size_background = X_b.size();
    int size_foreground = X_f.size();

    Tensor<double> phi_f_b(n_features, Matrix<double> (size_foreground, vector<double> (size_background, 0)));
    // For all foreground instances
    for (int i(0); i < size_foreground; i++){
        //cout << "Progress : " << (double) i / size_foreground << endl;
        // For all background instances
        for (int j(0); j < size_background; j++){
            // For all trees
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SAB(n_features+1, 0);
                vector<int> in_SA(n_features+1, 0);
                vector<double> phi(n_features, 0);

                // Start the recursion
                recurse(0, X_f[i], X_b[j], categorical_to_features, 
                        features[t], child_left[t], child_right[t],
                        threshold[t], value[t], W, n_features, phi, in_SAB, in_SA);

                for (int f(0); f < n_features; f++){
                    // Add the contribution to the sensitive attribute
                    phi_f_b[f][i][j] += phi[f];
                }
            }
        }
    }
    return phi_f_b;
}


int main(int argc, char **argv)
{
    
    // Load foreground and background
    Matrix<double> X_f = readFile<double>("foreground.txt");
    Matrix<double> X_b = readFile<double>("background.txt");

    // Load tree structure
    Matrix<int> categorical_to_features = readFile<int>("categorical_to_features.txt");
    Matrix<int> features = readFile<int>("feature.txt");
    Matrix<int> child_left  = readFile<int>("left.txt");
    Matrix<int> child_right = readFile<int>("right.txt");
    Matrix<double> threshold = readFile<double>("threshold.txt");
    Matrix<double> value = readFile<double>("value.txt");
    
    int n_features = categorical_to_features[categorical_to_features.size()-1][0] + 1;

    // Precompute the SHAP weights
    Matrix<double> W(n_features, vector<double> (n_features));
    compute_W(W);

    Tensor<double> Phi_f_b = treeSHAP(X_f, X_b, categorical_to_features,
                                        features, child_left, child_right,
                                        threshold, value, W);

    // This code is terribly bad... but it works :P
    cout << setprecision(20) << scientific;
    for (int f(0); f < n_features; f++){
        for (int i(0); i < X_f.size(); i++){
            for (int j(0); j < X_b.size()-1; j++){
                cout << Phi_f_b[f][i][j] << " ";
            }
            cout << Phi_f_b[f][i][X_b.size()-1] << endl;
        }
        cout << endl;
    }
}