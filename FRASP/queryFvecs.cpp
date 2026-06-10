#include "queryFRASP.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <set>
#include <cerrno>
#include <sstream>
#include <algorithm>
#include <utility> 

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

// num_lim <= 0: load all vectors in file; else load at most num_lim (for subsampling).
void readFVecs(const char *file_path, vector<float> &data, vector<int> &attributes, long long &num_vectors, long long &dimension, long long num_lim) {
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

    long long file_n = sz / row_sz;
    long long use_n = (num_lim > 0) ? std::min(file_n, num_lim) : file_n;
    num_vectors = use_n;

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
        attributes[i] = (int)i; 
    }

    fclose(f);
    cout << "Read vector data: " << num_vectors << " vectors of dimension " << dimension << endl;
}

vector<float> dt, queryData;

// query vector filename format: 4 bytes: query number; 4 bytes: dimension; query_nb*Dim vectors
void LoadQuery(std::string filename, long long &num_vectors, long long &dimension)
{
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        cerr << "Cannot open query file: " << filename << endl;
        return;
    }
    
    int nv;
    if (!infile.read((char *)&nv, sizeof(int))) {
        cerr << "Failed to read number of vectors" << endl;
        infile.close();
        return;
    }
    num_vectors = nv;
    
    int dim;
    if (!infile.read((char *)&dim, sizeof(int))) {
        cerr << "Failed to read dimension" << endl;
        infile.close();
        return;
    }
    dimension = dim;
    
    // 验证数据合理性
    if (num_vectors <= 0 || dimension <= 0) {
        cerr << "Invalid data: num_vectors=" << num_vectors << ", dimension=" << dimension << endl;
        infile.close();
        return;
    }
    
    queryData.resize(num_vectors * dimension);
    for (int i = 0; i < num_vectors; i++)
    {
        if (!infile.read((char *)queryData.data() + i * dimension * sizeof(float), dimension * sizeof(float))) {
            cerr << "Failed to read vector " << i << endl;
            infile.close();
            return;
        }
        // if(i > 990)
        //     cout << i*dimension << " " << queryData[i*dimension] << endl;
    }
    infile.close();
    cout << "Loaded " << num_vectors << " query vectors of dimension " << dimension << endl;
}

// Used only when computing groundtruth and constructing index. Do not use this to load data for search process
void LoadData(std::string filename, long long &num_vectors, long long &dimension)
{
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        cerr << "Cannot open data file: " << filename << endl;
        return;
    }
    
    int nv;
    if (!infile.read((char *)&nv, sizeof(int))) {
        cerr << "Failed to read number of vectors" << endl;
        infile.close();
        return;
    }
    num_vectors = nv;
    
    int dim;
    if (!infile.read((char *)&dim, sizeof(int))) {
        cerr << "Failed to read dimension" << endl;
        infile.close();
        return;
    }
    dimension = dim;
    
    // 验证数据合理性
    if (num_vectors <= 0 || dimension <= 0) {
        cerr << "Invalid data: num_vectors=" << num_vectors << ", dimension=" << dimension << endl;
        infile.close();
        return;
    }
    
    dt.resize(num_vectors * dimension);
    for (int i = 0; i < num_vectors; i++)
    {
        if (!infile.read((char *)dt.data() + i * dimension * sizeof(float), dimension * sizeof(float))) {
            cerr << "Failed to read vector " << i << endl;
            infile.close();
            return;
        }
    }
    infile.close();
    cout << "Loaded " << num_vectors << " data vectors of dimension " << dimension << endl;
}






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

void readBrute() {
    freopen(ground_truth_template, "r", stdin);
    for (int i = 0; i < q; i++) {
        string s; getline(cin, s);
        stringstream sin(s);
        int x;
        vector<int> now;
        while (sin >> x) {
            now.pb(x);
        }
        Br.pb(now);
    }
    fclose(stdin);
    
}

vector<FRASP::Query> Q;

void readGenerate() {
    freopen(query_path_template, "r", stdin);
     int l, r, i = 0; 
    while (~scanf("%d%d", &l, &r)) {
        if(l>r)
            std::swap(l,r);
        Q.pb({ l, r, i, i });
        i++;
    }
    
    fclose(stdin);
    cout << "Finish read query range" << endl;
    assert(Q.size() >= q);
    Q.resize(q);
}

std::unordered_map<std::string, std::string> paths;
int main(int argc, char **argv) {
    int segB = 1024, M = 0, ef = 0, resultK = 10, threads = 32;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_path"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_path"] = argv[i + 1];
        if (arg == "--index")
            paths["index"]=argv[i+1];
        if (arg == "--range_path")
            paths["range_path"] = argv[i + 1];
        if (arg == "--gt_path")
            paths["gt_path"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--ef")
            ef = std::stoi(argv[i + 1]);
        if (arg == "--B" || arg == "--K")
            segB = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            threads = std::stoi(argv[i + 1]);
        if (arg == "--result_k")
            resultK = std::stoi(argv[i + 1]);
    }
    if (paths["data_path"] == "")
    {
        cout<<("data path is empty");
        return 0;
    }   
    if (paths["query_path"] == "")
    {
        cout<<("query path is empty");
        return 0;
    }   
    if (paths["index"] == "")
    {
        cout<<("index path is empty");
        return 0;
    }   
    if (paths["range_path"] == "")
    {
        cout<<("range path is empty");
        return 0;
    }
    if (paths["gt_path"] == "")
    {
        cout<<("ground truth path is empty");
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
    if (segB <= 0)
    {
        cout<<("B (segment leaf threshold) should be a positive integer");
        return 0;
    }
    if (threads <= 0)
    {
        cout<<("threads should be a positive integer");
        return 0;
    }
    data_path_str = paths["data_path"];
    qdata_path_str = paths["query_path"];
    query_path_template_str = paths["range_path"];
    ground_truth_template_str = paths["gt_path"];


    data_path = data_path_str.c_str();
    qdata_path = qdata_path_str.c_str();
    query_path_template = query_path_template_str.c_str();
    ground_truth_template = ground_truth_template_str.c_str();

    cerr << "start!!!" << endl;
    long long baseNumVectors, baseDimension;
    readFVecs(data_path, dt, attributes, baseNumVectors, baseDimension, 0);

    long long queryNumVectors, queryDimension;
    vector<int> qat;
    readFVecs(qdata_path, queryData, qat, queryNumVectors, queryDimension, 0);
    // LoadQuery(qdata_path, queryNumVectors, queryDimension);
    // for(int i=2047999990;i<2048000000;i++)
    //     cout<<dt[i]<<" ";
    // cout<<endl;
    // cout<<queryNumVectors<<" "<<queryDimension<<endl;
    q = (int)queryNumVectors;
    if (baseDimension != queryDimension) {
        cerr << "base dim " << baseDimension << " != query dim " << queryDimension << endl;
        return 1;
    }
    cout << "M -> " << M << endl;
    cout << "B (leaf threshold) -> " << segB << endl;
    cout << "result_k -> " << resultK << endl;

    printMemoryUsage();

    FRASP u((int)baseNumVectors, (int)baseDimension, (int)queryNumVectors, M, ef, segB, resultK, threads, dt, queryData);
    readGenerate();
    readBrute();
    u.Br = Br;
    u.Q = Q;
    vector<vector<int>> predictedResults = u.run(paths["index"], 0, 0);
    puts("Finish");
    

    return 0;
}