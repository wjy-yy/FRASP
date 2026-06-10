#include "buildFRASP.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <set>
#include <sstream>
#include <algorithm>
#include <utility>
#include <sys/resource.h>




std::string v = "redcaps";

string pa = "../../../data/";

std::string data_path_str = pa + v + "/" + v + "_base.fvecs";
std::string qdata_path_str = pa + v + "/" + v + "_query.fvecs";
std::string query_path_template_str = pa + v + "/"  + "query.txt";
std::string ground_truth_template_str = pa + v + "/" + "ans.txt";


const char *data_path = data_path_str.c_str();
const char *qdata_path = qdata_path_str.c_str();
const char *query_path_template = query_path_template_str.c_str();
const char *ground_truth_template = ground_truth_template_str.c_str();

using namespace std;

void readFVecs(const char *file_path, vector<float> &data, vector<int> &attributes, long long &num_vectors, long long &dimension) {
    FILE* f = fopen(file_path, "rb");
    if (!f) {
        cerr << "Cannot open (vector file): " << file_path << endl;
        return;
    }

    uint32_t dim;
    if (fread(&dim, sizeof(dim), 1, f) != 1) {
        fclose(f);
        cerr << "Failed to read dimension" << endl;
        return;
    }
    dimension = dim;

    if (fseek(f, 0, SEEK_END) < 0) {
        fclose(f);
        cerr << "Failed to seek to end of file" << endl;
        return;
    }
    
    long long sz = ftello(f);
    if (sz < 0) {
        fclose(f);
        cerr << "Failed to get file size" << endl;
        return;
    }

    long long row_sz = (long long)sizeof(float) * dimension + sizeof(int);
    if (sz % row_sz != 0) {
        fclose(f);
        cerr << "Invalid file size" << endl;
        return;
    }

    num_vectors = sz / row_sz;

    data.resize(num_vectors * dimension);
    attributes.resize(num_vectors);
    if (fseek(f, 0, SEEK_SET) < 0) {
        fclose(f);
        cerr << "Failed to seek to start of file" << endl;
        return;
    }

    for (long long i = 0; i < num_vectors; i++) {
        int dim_;
        if (fread(&dim_, sizeof(int), 1, f) != 1 || 
            fread(data.data() + i * dimension, sizeof(float), dimension, f) != dimension) {
            fclose(f);
            cerr << "Failed to read vector data" << endl;
            return;
        }
        attributes[i] = i;
    }

    fclose(f);
    cout << "Read vector data: " << num_vectors << " vectors of dimension " << dimension << endl;
}

vector<float> dt, queryData;




double calculateRecall(const vector<vector<int>>& predicted, const vector<vector<int>>& groundTruth) {
    double totalRecall = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        const auto& pred = predicted[i];
        const auto& gt = groundTruth[i];

        set<int> gtSet(gt.begin(), gt.end());
        int truePositives = 0;

        for (int p : pred) {
            if (gtSet.count(p)) {
                truePositives++;
            }
        }

        double recall = static_cast<double>(truePositives) / gt.size();
        totalRecall += recall;
    }
    return totalRecall / predicted.size();
}

int q;

vector<vector<int> > Br;
vector<int> attributes;

vector<FRASP::Query> Q;

std::unordered_map<std::string, std::string> paths;
int main(int argc, char **argv) {
    int M = 0, ef = 0, threads = 0, B = 10;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_path"]=argv[i+1];
        if (arg == "--index_file")
            paths["index_file"]=argv[i+1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--ef")
            ef = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            threads = std::stoi(argv[i + 1]);
        if (arg == "--B" || arg == "--K")
            B = std::stoi(argv[i + 1]);
    }
    if (paths["data_path"] == "")
    {
        cout<<("data path is empty");
        return 0;
    }
    if (paths["index_file"] == "")
    {
        cout<<("index path is empty");
        return 0;
    }
    if (M <= 0)
    {
        cout<<("M should be a positive integer");
        return 0;
    }
    if (ef <= 0)
    {
        cout<<("ef_construction should be a positive integer");
        return 0;
    }
    if (threads <= 0)
    {
        cout<<("threads should be a positive integer");
        return 0;
    }

    data_path_str = paths["data_path"];

    data_path = data_path_str.c_str();

    long long baseNumVectors, baseDimension;
    readFVecs(data_path, dt, attributes, baseNumVectors, baseDimension);
    
    cout << "M -> " << M << endl;
    cout << "B (leaf threshold) -> " << B << endl;

    printMemoryUsage();
    FRASP u(baseNumVectors, baseDimension, M, ef, B, threads, dt);
    u.run(paths["index_file"]); 

    return 0;
}