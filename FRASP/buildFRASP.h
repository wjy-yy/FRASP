#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <set>
#include <iomanip>
#include <mutex>
#include <chrono>
#include <ctime>
#include "hnswlib.h"
#include <atomic>
#include "memory.hpp"
#include "ThreadPool.h"
#include <sys/resource.h>
#include <sys/time.h>

using namespace std;
using namespace hnswlib;
float GetTime(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC;
}
typedef pair<int, int> PII;
inline double rssGb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0 / 1024.0;
}

void printMemoryUsage() {
    std::cout << "Memory usage: " << rssGb() << " GB" << std::endl;
}


struct FRASP {

    int n, D, M, ef, K, maxThreads; 
    vector<float> &dt;
    
    FRASP(int nn, int dd, int mm, int eff, int kk, int mxthread, vector<float> &DT)
        : n(nn), D(dd), M(mm), ef(eff), K(kk), maxThreads(mxthread), dt(DT) {
    }


    HierarchicalNSW<float>* H;
    vector<HierarchicalNSW<float>*> L, R;

    void add(HierarchicalNSW<float> *hnsw, int x, int i) {
        hnsw->addPoint((void*)(dt.data() + x * D), i);
    }

    #define ls (p << 1)
    #define rs (p << 1 | 1)
    #define pb push_back

    struct Query{
        int l, r, u, id;
    };

    vector<Query> Q;
    vector<vector<int> > ans;

    double buildTime;

    struct Tree{
        int p, l, r;
    };

    vector<Tree> TN;

    std::atomic<int> hnsw_built_{0};
    std::atomic<size_t> total_index_bytes_{0};
    int hnsw_total_expected_{0};
    std::mutex build_log_mu_;

    void onHnswBuilt(const std::string &tag, HierarchicalNSW<float> *h, int points) {
        const size_t ib = h ? h->indexFileSize() : 0;
        const int done = ++hnsw_built_;
        total_index_bytes_.fetch_add(ib);
        std::lock_guard<std::mutex> lk(build_log_mu_);
        std::cerr << std::fixed << std::setprecision(2)
                  << "[build] " << tag << " done"
                  << "  points=" << points
                  << "  this_index=" << (ib / 1024.0 / 1024.0) << " MB"
                  << "  cumulative_index=" << (total_index_bytes_.load() / 1024.0 / 1024.0 / 1024.0) << " GB"
                  << "  built=" << done << "/" << hnsw_total_expected_
                  << "  RSS=" << rssGb() << " GB\n"
                  << std::flush;
    }

    void wk(int p, int l, int r, L2Space& space) {
        if (l == r) return;
        int mid = (l + r) >> 1;

        int z = mid - l + 1;
        
        L[p] = new HierarchicalNSW<float>(&space, z, M, ef);

        for (int i = 0; i < z; i++) {
            add(L[p], mid - i, i);
        }
        L[p] -> st.mid = mid - ((l + mid) / 2 + 1);
        L[p] -> prework();
        {
            std::ostringstream oss;
            oss << "L[p=" << p << " range=" << l << "," << mid << "]";
            onHnswBuilt(oss.str(), L[p], z);
        }

        z = r - mid;

        R[p] = new HierarchicalNSW<float>(&space, z, M, ef);

        for (int i = 0; i < z; i++) {
            add(R[p], mid + 1 + i, i);
        }
        R[p] -> st.mid =  (r + mid + 1) / 2 - (mid + 1);
        R[p] -> prework();
        {
            std::ostringstream oss;
            oss << "R[p=" << p << " range=" << mid + 1 << "," << r << "]";
            onHnswBuilt(oss.str(), R[p], z);
        }
    }

   

    

    void sv(int p, int l, int r) {
        if (r - l + 1 <= K) return;
        int mid = (l + r) >> 1;
        TN.pb({ p, l, r });

        sv(ls, l, mid);
        sv(rs, mid + 1, r);
    }




    typedef pair<float, int> PFI;

    struct TH{
        int l, r, k, b;
        HierarchicalNSW<float> *h;
    };



     hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};
    
    

    inline void *getDataByInternalId(int x) const
    {
        return (void*)(dt.data() + x * D);
    }

    void buildH(L2Space& space) {
        H = new HierarchicalNSW<float>(&space, n, M, ef);

        const long long step = std::max(1LL, static_cast<long long>(n) / 200);
        std::cerr << "[buildH] inserting " << n << " points into global HNSW (M=" << M << ", ef=" << ef
                  << "), progress ~every " << step << " points\n"
                  << std::flush;
        for (int i = 0; i < n; i++) {
            add(H, i, i);
            const long long done = static_cast<long long>(i) + 1;
            if (done == 1LL || done == static_cast<long long>(n) || (done % step == 0)) {
                std::cerr << "[buildH] " << done << " / " << n << "  ("
                          << std::fixed << std::setprecision(2)
                          << (100.0 * static_cast<double>(done) / static_cast<double>(n)) << "%)\n"
                          << std::flush;
            }
        }
        H->st.mid = ((0 + n - 1) / 2);
        H->prework();
        std::cerr << "[buildH] prework done\n" << std::flush;
        onHnswBuilt("H[global 0..n-1]", H, n);
    }

    void saveData(std::ofstream &out, const std::vector<int>& data) {
       
        size_t size = data.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(reinterpret_cast<const char*>(data.data()), size * sizeof(int));
       
    }

    void saveData(std::ofstream &out, const std::vector<std::vector<int>>& data) {
        
        size_t outerSize = data.size();
        out.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));
        for (const auto& vec : data) {
            size_t innerSize = vec.size();
            out.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
            out.write(reinterpret_cast<const char*>(vec.data()), innerSize * sizeof(int));
        }
    }

    void saveData(std::ofstream &out, const std::vector<std::vector<std::vector<std::pair<int, int>>>>& data) {
    
        size_t outerSize = data.size();
        out.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));
        for (const auto& matrix : data) {
            size_t midSize = matrix.size();
            out.write(reinterpret_cast<const char*>(&midSize), sizeof(midSize));
            for (const auto& vec : matrix) {
                size_t innerSize = vec.size();
                out.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
                out.write(reinterpret_cast<const char*>(vec.data()), innerSize * sizeof(std::pair<int, int>));
            }
        }
    }

    void saveData(std::ofstream &out, const std::vector<std::vector<std::vector<int>>>& data) {
    
        size_t outerSize = data.size();
        out.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));
        for (const auto& matrix : data) {
            size_t midSize = matrix.size();
            out.write(reinterpret_cast<const char*>(&midSize), sizeof(midSize));
            for (const auto& vec : matrix) {
                size_t innerSize = vec.size();
                out.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
                out.write(reinterpret_cast<const char*>(vec.data()), innerSize * sizeof(int));
            }
        }
    }
    
    void run(const string& path) {
        
        
        cerr << "N : " << n << endl;
        cerr << "EF build : " << ef << endl;
        cerr << "B (leaf threshold K) : " << K << endl;

        L.resize(4 * n + 1, nullptr);
        R.resize(4 * n + 1, nullptr);

        L2Space space(D);
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();
        timeval t1, t2;
        gettimeofday(&t1, NULL);
        sv(1, 0, n - 1);
        hnsw_total_expected_ = 1 + 2 * static_cast<int>(TN.size());
        cerr << "[build] segment-tree nodes=" << TN.size()
             << "  total HNSW graphs=" << hnsw_total_expected_
             << "  (1 global + 2 per node)\n"
             << std::flush;

        ThreadPool pool(maxThreads);
        std::vector< std::future<void> > results;


        results.emplace_back(
            pool.enqueue([=, &space]() {
                this->buildH(space);
            })
        );
        printMemoryUsage();
    
        
        for (auto t : TN) {
            results.emplace_back(
                pool.enqueue([=, &space]() {
                    wk(t.p, t.l, t.r, space);
                })
            );
        }

        for (auto& result : results) {
            result.get();
        }
        gettimeofday(&t2, NULL);
        double construction_time = GetTime(t1, t2);

        std::cout << "Build time:" << construction_time << "s" << std::endl;
        puts("Finish Build");
         printMemoryUsage();
        
        std::ofstream output(path, std::ios::binary);
        for(int i=0;i<H->ed.size();i++)
            for(int j=0;j<H->ed[i].size();j++)
                sort(H->ed[i][j].begin(),H->ed[i][j].end());
        saveData(output, H -> st.a);
        saveData(output, H -> st.pr);
        saveData(output, H -> st.f);
        saveData(output, H -> ed);
        saveData(output, H -> del);
        for (size_t i = 0; i < L.size(); ++i) {
             if (L[i]) {
                for(int _=0;_<L[i]->ed.size();_++)
                    for(int __=0;__<L[i]->ed[_].size();__++)
                    {
                        sort(L[i]->ed[_][__].begin(),L[i]->ed[_][__].end());
                    }
                saveData(output, L[i] -> st.a);
                saveData(output, L[i] -> st.pr);
                saveData(output, L[i] -> st.f);
                saveData(output, L[i] -> ed);
                saveData(output, L[i] -> del);
                
            }
            if (R[i]) {
                for(int _=0;_<R[i]->ed.size();_++)
                    for(int __=0;__<R[i]->ed[_].size();__++)
                    {
                        sort(R[i]->ed[_][__].begin(),R[i]->ed[_][__].end());            
                    }
                saveData(output, R[i] -> st.a);
                saveData(output, R[i] -> st.pr);
                saveData(output, R[i] -> st.f);
                saveData(output, R[i] -> ed);
                saveData(output, R[i] -> del);
                
            }
        }

        

        cout << "HNSW saved to " + path << endl;
        output.close();

        size_t index_bytes = 0;
        if (std::filesystem::exists(path))
            index_bytes = std::filesystem::file_size(path);

        const std::string build_log = path + ".build.log";
        {
            std::ofstream bl(build_log);
            const auto now = std::chrono::system_clock::now();
            const std::time_t t = std::chrono::system_clock::to_time_t(now);
            bl << "finished_at=" << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "\n";
            bl << "index_file=" << path << "\n";
            bl << "n=" << n << " D=" << D << " M=" << M << " ef=" << ef << " B=" << K
               << " threads=" << maxThreads << "\n";
            bl << "construction_time_sec=" << std::fixed << std::setprecision(3)
               << construction_time << "\n";
            bl << "peak_rss_gb=" << std::setprecision(2) << rssGb() << "\n";
            bl << "index_bytes=" << index_bytes << "\n";
            bl << "index_size_gb=" << std::setprecision(3)
               << (index_bytes / 1024.0 / 1024.0 / 1024.0) << "\n";
            bl << "hnsw_graphs_built=" << hnsw_built_.load() << "\n";
        }
        std::cerr << "[build] DONE total_time=" << construction_time << "s"
                  << " index_size=" << (index_bytes / 1024.0 / 1024.0 / 1024.0) << " GB"
                  << " log=" << build_log << "\n"
                  << std::flush;
    }
};
