#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <sys/time.h>
#include <set>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include "memory.hpp"
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <sys/resource.h>
#include <cstdio>
#include <cstring>

// (removed parallel cache helpers; single-threaded only)

void printMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory usage: " << usage.ru_maxrss / 1024.0 / 1024.0 << " GB" << std::endl;
}

using namespace std;
using namespace hnswlib;

namespace {

bool frasp_env_flag_on(const char *name) {
    const char *v = std::getenv(name);
    return v && v[0] == '1';
}

// Figure 11 ablation: straddling queries that use parent + L[p] + R[p] together.
// 0 = full FRASP (default), 1 = child-only (L/R only), 2 = parent-only (cross-range HNSW only).
inline int frasp_ablation_mode() {
    const char *s = std::getenv("FRASP_ABLATION");
    if (!s || !*s) return 0;
    if (std::strcmp(s, "child") == 0 || std::strcmp(s, "1") == 0) return 1;
    if (std::strcmp(s, "parent") == 0 || std::strcmp(s, "2") == 0) return 2;
    if (std::strcmp(s, "full") == 0 || std::strcmp(s, "0") == 0) return 0;
    return 0;
}

// FRASP_ASSERT_ED_DEGREES=1 or legacy FRASP_ASSERT_MAX_ED0_DEG=1:
// abort if any ed[lev][v] exceeds HNSW caps (layer0: maxM0_=2*M, upper: maxM_=M).
bool frasp_assert_ed_degree_caps() {
    return frasp_env_flag_on("FRASP_ASSERT_ED_DEGREES")
        || frasp_env_flag_on("FRASP_ASSERT_MAX_ED0_DEG");
}

// Returns max |ed[0][v]| for logging; optionally abort on violation.
size_t frasp_check_hnsw_ed_degrees(HierarchicalNSW<float> *h, const char *name,
                                   bool abort_on_violation) {
    if (!h) return 0;
    size_t mx0 = 0;
    const int nlev = h->ed_num_levels();
    for (int lev = 0; lev < nlev; ++lev) {
        const size_t cap = (lev == 0) ? h->maxM0_ : h->maxM_;
        const int nv = h->ed_num_vertices(lev);
        for (int v = 0; v < nv; ++v) {
            auto [eb, ee] = h->ed_neighbors(lev, v);
            const size_t d = static_cast<size_t>(ee - eb);
            if (lev == 0) mx0 = std::max(mx0, d);
            if (d > cap) {
                std::cerr << "FRASP ed degree: " << name << " ed[" << lev << "][" << v
                          << "].size()=" << d << " > cap " << cap
                          << " (layer0 cap=maxM0_=" << h->maxM0_
                          << ", upper cap=maxM_=" << h->maxM_ << ")\n";
                if (abort_on_violation) std::abort();
            }
        }
    }
    return mx0;
}

static uint64_t frasp_count_ed_edges(const HierarchicalNSW<float> *h) {
    if (!h) return 0;
    uint64_t tot = 0;
    if (h->ed_use_csr()) {
        for (const auto &pool : h->ed_csr_edges_)
            tot += pool.size();
    } else {
        for (const auto &level : h->ed)
            for (const auto &adj : level)
                tot += adj.size();
    }
    return tot;
}

static uint64_t frasp_count_del_edges(const HierarchicalNSW<float> *h) {
    if (!h) return 0;
    uint64_t tot = 0;
    if (h->del_use_csr()) {
        for (const auto &pool : h->del_csr_edges_)
            tot += pool.size();
    } else {
        for (const auto &level : h->del)
            for (const auto &adj : level)
                tot += adj.size();
    }
    return tot;
}

static void frasp_release_query_scaffold(HierarchicalNSW<float> *h) {
    if (!h) return;
    h->vis.clear();
    h->vis.shrink_to_fit();
    h->element_levels_.clear();
    h->element_levels_.shrink_to_fit();
}

static void frasp_skip_int_vector_blob(std::ifstream &in) {
    size_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(size));
    if (size)
        in.seekg(static_cast<std::streamoff>(size * sizeof(int)), std::ios::cur);
}

static void frasp_load_ed_as_csr(std::ifstream &in, HierarchicalNSW<float> &h) {
    size_t outerSize = 0;
    in.read(reinterpret_cast<char *>(&outerSize), sizeof(outerSize));
    h.ed.clear();
    h.ed.shrink_to_fit();
    h.ed_csr_off_.clear();
    h.ed_csr_edges_.clear();
    h.ed_csr_off_.resize(outerSize);
    h.ed_csr_edges_.resize(outerSize);
    for (size_t lev = 0; lev < outerSize; ++lev) {
        size_t midSize = 0;
        in.read(reinterpret_cast<char *>(&midSize), sizeof(midSize));
        auto &off = h.ed_csr_off_[lev];
        auto &pool = h.ed_csr_edges_[lev];
        off.resize(midSize + 1);
        off[0] = 0;
        uint32_t cursor = 0;
        for (size_t u = 0; u < midSize; ++u) {
            size_t innerSize = 0;
            in.read(reinterpret_cast<char *>(&innerSize), sizeof(innerSize));
            if (innerSize) {
                const size_t old = pool.size();
                pool.resize(old + innerSize);
                in.read(reinterpret_cast<char *>(pool.data() + old),
                        innerSize * sizeof(int));
                std::sort(pool.begin() + static_cast<std::ptrdiff_t>(old), pool.end());
                cursor += static_cast<uint32_t>(innerSize);
            }
            off[u + 1] = cursor;
        }
    }
}

static void frasp_load_del_as_csr(std::ifstream &in, HierarchicalNSW<float> &h) {
    size_t outerSize = 0;
    in.read(reinterpret_cast<char *>(&outerSize), sizeof(outerSize));
    h.del.clear();
    h.del.shrink_to_fit();
    h.del_csr_off_.clear();
    h.del_csr_edges_.clear();
    h.del_csr_off_.resize(outerSize);
    h.del_csr_edges_.resize(outerSize);
    for (size_t lev = 0; lev < outerSize; ++lev) {
        size_t midSize = 0;
        in.read(reinterpret_cast<char *>(&midSize), sizeof(midSize));
        auto &off = h.del_csr_off_[lev];
        auto &pool = h.del_csr_edges_[lev];
        off.resize(midSize + 1);
        off[0] = 0;
        uint32_t cursor = 0;
        for (size_t u = 0; u < midSize; ++u) {
            size_t innerSize = 0;
            in.read(reinterpret_cast<char *>(&innerSize), sizeof(innerSize));
            if (innerSize) {
                const size_t old = pool.size();
                pool.resize(old + innerSize);
                in.read(reinterpret_cast<char *>(pool.data() + old),
                        innerSize * sizeof(std::pair<int, int>));
                cursor += static_cast<uint32_t>(innerSize);
            }
            off[u + 1] = cursor;
        }
    }
}

static void frasp_add_graph_edge_counts(HierarchicalNSW<float> *g,
                                        uint64_t &ed, uint64_t &del) {
    ed += frasp_count_ed_edges(g);
    del += frasp_count_del_edges(g);
}

static void frasp_print_index_edge_stats(
    HierarchicalNSW<float> *H,
    const std::vector<HierarchicalNSW<float> *> &L,
    const std::vector<HierarchicalNSW<float> *> &R,
    const std::vector<int> &ND) {
    uint64_t ed_sum = 0, del_sum = 0;
    frasp_add_graph_edge_counts(H, ed_sum, del_sum);
    for (size_t i = 0; i < L.size(); ++i) {
        if (!ND[i]) continue;
        if (L[i]) frasp_add_graph_edge_counts(L[i], ed_sum, del_sum);
        if (i < R.size() && R[i]) frasp_add_graph_edge_counts(R[i], ed_sum, del_sum);
    }
    const uint64_t total_edges = ed_sum + del_sum;
    const double pruned_m = del_sum / 1e6;
    const double total_m = total_edges / 1e6;
    const double ratio_pct = total_edges ? 100.0 * del_sum / total_edges : 0.0;
    std::cerr << "FRASP_INDEX_EDGE_STATS"
              << "  active_edges_ed=" << ed_sum
              << "  pruned_edges_del=" << del_sum
              << "  total_edges=" << total_edges
              << "  active_x1e6=" << (ed_sum / 1e6)
              << "  pruned_x1e6=" << pruned_m
              << "  total_x1e6=" << total_m
              << "  ratio_pct=" << ratio_pct
              << "  (total = |ed|+|del| over H + all loaded L/R, all layers)\n";
}

}  // namespace

std::ofstream ouf("gist2out.txt");
typedef pair<int, int> PII;

// ================================================================
// 优化1: FastBitset —— visited 位图。曾用 dirty_buf 只清「脏块」且 cap=8192；
// n=1e6 时块数约 15625，高 EF 访问块数多时超出 cap 的块 reset() 清不到 → 后续查询误判已访问。
// 现 reset() 整表 memset（每查询 ~nblocks*8B，可接受），保证正确性。
// ================================================================
template <typename Block = uint64_t>
struct FastBitset {
private:
    static constexpr int block_size = sizeof(Block) * 8;
    int nblocks;
    Block *data;

public:
    explicit FastBitset(int n)
        : nblocks((n + block_size - 1) / block_size),
          data(static_cast<Block*>(memory::align_mm<64>(nblocks * sizeof(Block))))
    {
        std::memset(data, 0, nblocks * sizeof(Block));
    }

    ~FastBitset() { free(data); }

    inline void reset() { std::memset(data, 0, nblocks * sizeof(Block)); }

    inline void set(int i) {
        int bi = i / block_size;
        Block &slot = data[bi];
        Block mask = Block(1) << (i & (block_size - 1));
        slot |= mask;
    }

    inline bool get(int i) {
        return (data[i / block_size] >> (i & (block_size - 1))) & 1;
    }

    inline bool test_and_set(int i) {
        int bi = i / block_size;
        Block mask = Block(1) << (i & (block_size - 1));
        Block &slot = data[bi];
        if (slot & mask) return false;
        slot |= mask;
        return true;
    }

    void *block_address(int i) { return data + i / block_size; }
};

// define thread_local static members
 

struct FRASP {

    int n, D, q, M, ef, K, maxThreads;
    int segB;  // segment-tree leaf threshold (must match index build --B)
    int B, dcnt{0};
    /// Query-time only (not index M): after getEdgeFastBinary merges all th[] range slices into neighbor_buf,
    /// only the first N entries get a full distance / heap update. 0 = no limit.
    /// This can exceed 2*M because (1) multiple subgraphs are unioned, (2) loaded ed[0] may have high degree
    /// after the on-disk pairwise expansion in run() — see cerr max_degree after index load.
    int getedge_max_neighbors_per_pop_{0};
    vector<float> &dt, &queryData;
    FRASP(int nn, int dd, int qq, int mm, int eff, int leaf_b, int result_k, int mxthread,
          vector<float> &DT, vector<float> &QDT)
        : n(nn), D(dd), q(qq), M(mm), ef(eff), K(result_k), segB(leaf_b), maxThreads(mxthread),
          dt(DT), queryData(QDT) {}

    size_t data_size_, prefetch_lines;
    HierarchicalNSW<float>* H;
    vector<HierarchicalNSW<float>*> L, R;

    void add(HierarchicalNSW<float> *hnsw, int x, int i) {
        hnsw->addPoint((void*)(dt.data() + x * D), i);
    }

#define ls (p << 1)
#define rs (p << 1 | 1)
#define pb push_back

    struct Query { int l, r, u, id; };
    vector<Query> Q;
    vector<vector<int>> ans;
    double queryTime;

    void wk(int p, int l, int r, L2Space& space) {
        if (l == r) return;
        int mid = (l + r) >> 1;
        int z = mid - l + 1;
        L[p] = new HierarchicalNSW<float>(&space, z, M, ef);
        for (int i = 0; i < z; i++) add(L[p], mid - i, i);
        L[p]->buildST();
        z = r - mid;
        R[p] = new HierarchicalNSW<float>(&space, z, M, ef);
        for (int i = 0; i < z; i++) add(R[p], mid + 1 + i, i);
        R[p]->buildST();
    }

    struct Tree { int p, l, r; };
    vector<Tree> TN;

    void sv(int p, int l, int r) {
        if (r - l + 1 <= segB) return;
        int mid = (l + r) >> 1;
        TN.pb({ p, l, r });
        sv(ls, l, mid);
        sv(rs, mid + 1, r);
    }

    typedef pair<float, int> PFI;

    struct TH {
        int l, r, k, b;
        HierarchicalNSW<float> *h;
    };

    struct THArr {
        TH data[4];
        int sz{0};
        inline void clear() { sz = 0; }
        inline void pb(TH t) { data[sz++] = t; }
        inline int size() const { return sz; }
        inline TH& operator[](int i) { return data[i]; }
        inline const TH& operator[](int i) const { return data[i]; }
    };

    /// Qry 递归时复用（单线程 svQuery）；避免每层栈上 THArr + 反复默认构造。
    THArr th_scratch_{};

    float getD(int v, int u) {
        float d = 0; int px = v*D, py = u*D;
        for (int i = 0; i < D; i++) { float dx = dt[px+i]-dt[py+i]; d += dx*dx; }
        return d;
    }

    hnswlib::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};

    // ================================================================
    // [NEW 优化X] per-query distance cache (timestamped) to avoid
    // recomputing distances to the same internal id repeatedly.
    // Memory: ~8 bytes * n (float + uint32) -> acceptable for 1e6 points.
    // ================================================================
    std::vector<float> dist_cache;
    std::vector<uint32_t> dist_cache_ts;
    uint32_t dist_cur_ts{1};
    bool enable_dist_cache{false};  // Disable cache - adds overhead

    inline float getDistCached(void *query_data, int internalId) {
        if (!enable_dist_cache) {
            ++dcnt;
            return fstdistfunc_(query_data, getDataByInternalId(internalId), dist_func_param_);
        }
        uint32_t id = static_cast<uint32_t>(internalId);
        if (dist_cache_ts[id] != dist_cur_ts) {
            ++dcnt;
            const float d =
                fstdistfunc_(query_data, getDataByInternalId(internalId), dist_func_param_);
            dist_cache[id] = d;
            dist_cache_ts[id] = dist_cur_ts;
            return d;
        }
        return dist_cache[id];
    }

    inline void *getDataByInternalId(int x) const {
        return (void*)(dt.data() + x * D);
    }

    static constexpr int NEIGHBOR_BUF_SIZE = 8192;
    int neighbor_buf[NEIGHBOR_BUF_SIZE];

    // ================================================================
    // [NEW 优化6] 预分配的 heap 存储 — 跨查询复用
    //   原代码：每次 query() 都新建 priority_queue，底层 vector 堆分配
    //   现在：复用两个 vector，用 push_heap/pop_heap 手动管理
    // ================================================================
    vector<PFI> can_storage;   // min-heap (greater)
    vector<PFI> tp_storage;    // max-heap (less)

    // ================================================================
    // 优化3: getEdgeFastBinary (保留)
    // ================================================================
    inline int getEdgeFastBinary(int u, int ql, int qr, const THArr &th,
                                  FastBitset<uint64_t> &vis) {
        int cnt = 0;
        for (int i = 0; i < th.sz; i++) {
            const auto &o = th[i];
            if (u < o.l || u > o.r) continue;

            const int pu = (o.k == 1) ? (u - o.b) : (o.b - u);

            int lo_lim, hi_lim;
            if (o.k == 1) {
                lo_lim = ql - o.b; if (lo_lim < 0) lo_lim = 0;
                hi_lim = qr - o.b;
            } else {
                lo_lim = o.b - qr; if (lo_lim < 0) lo_lim = 0;
                hi_lim = o.b - ql;
            }

            if (pu < o.h->del_num_vertices(0)) {
                auto [db, de] = o.h->del_neighbors(0, pu);
                for (const auto *it = db; it != de; ++it) {
                    const auto &ne = *it;
                    if (ne.first > hi_lim) break;
                    if (ne.first < lo_lim) continue;
                    if (ne.second > hi_lim) {
                        const int v = (o.k == 1) ? (ne.first + o.b) : (o.b - ne.first);
                        if (vis.test_and_set(v)) {
                            if (cnt + 1 < NEIGHBOR_BUF_SIZE)
                                __builtin_prefetch(getDataByInternalId(v), 0, 1);
                            neighbor_buf[cnt++] = v;
                        }
                    }
                }
            }

            if (pu < o.h->ed_num_vertices(0)) {
                auto [eb, ee] = o.h->ed_neighbors(0, pu);
                if (eb == ee) continue;
                auto it = eb;
                if(lo_lim>*eb)
                    it = std::lower_bound(eb, ee, lo_lim);
                for (; it != ee; ++it) {
                    const int ne = *it;
                    if (ne > hi_lim) break;
                    const int v = (o.k == 1) ? (ne + o.b) : (o.b - ne);
                    if (vis.test_and_set(v)) {
                        if (cnt + 1 < NEIGHBOR_BUF_SIZE)
                            __builtin_prefetch(getDataByInternalId(v), 0, 1);
                        neighbor_buf[cnt++] = v;
                    }
                }
            }
        }
        return cnt;
    }

    int getCur(HierarchicalNSW<float>* h, void *query_data, size_t k,
               int K, int B, int l, int r, int id) {
        int x = (l - B) / K, y = (r - B) / K;
        if (x > y) std::swap(x, y);
        int currObj = h->st.query(x, y);

        float curdist = getDistCached(query_data, currObj * K + B);
        int nl = h->st.a[currObj];
        
        for (int level = nl; level > 0; level--) {
            bool changed = true;
            
            while (changed) {
                changed = false;
                if (currObj < h->ed_num_vertices(level)) {
                    auto [eb, ee] = h->ed_neighbors(level, currObj);
                    for (const int *it = eb; it != ee; ++it) {
                        const int cand = *it;
                        const int ne = cand * K + B;
                        if (ne < l || ne > r) continue;
                        float d = getDistCached(query_data, ne);
                        if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                    }
                }
                if (currObj < h->del_num_vertices(level)) {
                    auto [db, de] = h->del_neighbors(level, currObj);
                    for (const auto *it = db; it != de; ++it) {
                        const auto &o = *it;
                        tableint cand = o.first;
                        if (cand > y) break;
                        int ne = cand * K + B;
                        if (ne < l || ne > r) continue;
                        if (o.second > y) {
                            float d = getDistCached(query_data, ne);
                            if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                        }
                    }
                }
            }
        }
        return currObj;
    }

    // ================================================================
    // [NEW 优化7] query() 改造：
    //   - 复用 can_storage / tp_storage（跨查询）
    //   - 默认仅用 th[0] 作入口 getCur（其余靠 getEdgeFastBinary 在 th[*] 上扩边）
    //   - child-only（FRASP_ABLATION=child）：th 只有 disconnected 子图，必须对每个 th[i] 都做 getCur，
    //     否则只有一个入口只会从一侧启动搜索（Fig.11 ablation 语义）
    // ================================================================
    void query(Query qr, int l, int mid, int r,
               THArr &th, FastBitset<uint64_t> &vis, vector<int> &out) {
        auto query_data = (void*)(queryData.data() + qr.u * D);

        // 复用底层存储：清零 size，不释放 capacity
        can_storage.clear();
        tp_storage.clear();

        const int ab = frasp_ablation_mode();
        const int n_entry = (ab == 1) ? th.sz : 1;
        for (int ti = 0; ti < n_entry; ++ti) {
            auto &u = th[ti];
            int le = (qr.l > u.l) ? qr.l : u.l;
            int re = (qr.r < u.r) ? qr.r : u.r;
            const int pid = getCur(u.h, query_data, K, u.k, u.b, le, re, qr.id) * u.k + u.b;
            if (vis.test_and_set(pid)) {
                float dis = getDistCached(query_data, pid);
                can_storage.emplace_back(dis, pid);
                tp_storage.emplace_back(dis, pid);
                std::push_heap(can_storage.begin(), can_storage.end(), greater<PFI>());
                std::push_heap(tp_storage.begin(), tp_storage.end());
            }
        }

        if (tp_storage.empty()) { out.clear(); return; }

        float lowerBound = tp_storage.front().first;

        while (!can_storage.empty()) {
            PFI cur = can_storage.front();
            if (cur.first > lowerBound && (int)tp_storage.size() >= ef) break;
            std::pop_heap(can_storage.begin(), can_storage.end(), greater<PFI>());
            can_storage.pop_back();

            const int u = cur.second;
            const int num = getEdgeFastBinary(u, qr.l, qr.r, th, vis);

            for (int i = 0; i < num; ++i) {
                const int v = neighbor_buf[i];
                const float dis = getDistCached(query_data, v);
                const int tsz = (int)tp_storage.size();
                if (tsz < ef || dis < lowerBound) {
                    can_storage.emplace_back(dis, v);
                    std::push_heap(can_storage.begin(), can_storage.end(), greater<PFI>());
                    tp_storage.emplace_back(dis, v);
                    std::push_heap(tp_storage.begin(), tp_storage.end());
                    if ((int)tp_storage.size() > ef) {
                        std::pop_heap(tp_storage.begin(), tp_storage.end());
                        tp_storage.pop_back();
                    }
                    lowerBound = tp_storage.front().first;
                }
            }
        }

        // 取前 K：tp_storage 是 max-heap，pop K 次得到最大的（弹出非 K 的）
        while ((int)tp_storage.size() > K) {
            std::pop_heap(tp_storage.begin(), tp_storage.end());
            tp_storage.pop_back();
        }
        out.clear();
        out.reserve(tp_storage.size());
        // 直接拷贝；getRecall 内会按到 query 的距离排序再做 rank-aligned d_err
        for (auto &p : tp_storage) out.pb(p.second);
    }

    // [优化8+] 复用 th_scratch_，减少 Qry 每层栈上 THArr 与 clear 成本（单线程查询路径）。
    void Qry(int p, int l, int r, Query u, FastBitset<uint64_t> &vis, vector<int> &out) {
        THArr &th = th_scratch_;
        th.clear();
        const int mid = (l + r) >> 1;
        if (u.r == mid) {
            th.pb((TH){ l, mid, -1, mid, L[p] });
            query(u, l, mid, r, th, vis, out);
            return;
        }
        if (u.l == mid + 1) {
            th.pb((TH){ mid+1, r, 1, mid+1, R[p] });
            query(u, l, mid, r, th, vis, out);
            return;
        }
        if (u.l <= mid && mid < u.r) {
            int f = p >> 1;
            const int ab = frasp_ablation_mode();
            // [优化7] 把 parent 放在 th[0]（full 模式）；child-only 仅用 L/R，parent-only 仅用跨区间 parent
            if (ab != 1) {
                if (f) {
                    if ((f << 1) == p)
                        th.pb((TH){ l, r, -1, r, L[f] });
                    else
                        th.pb((TH){ l, r, 1, l, R[f] });
                } else {
                    th.pb((TH){ l, r, 1, l, H });
                }
            }
            if (ab != 2) {
                th.pb((TH){ l, mid, -1, mid, L[p] });
                th.pb((TH){ mid+1, r, 1, mid+1, R[p] });
            }
            query(u, l, mid, r, th, vis, out);
            return;
        }
        if (u.r <= mid) Qry(ls, l, mid, u, vis, out);
        else if (u.l > mid) Qry(rs, mid+1, r, u, vis, out);
        else assert(false);
    }

    // ================================================================
    // [NEW 优化9] svQuery：预 reserve 堆容量 + 直接写入 ans[u.id]
    // ================================================================
    auto svQuery() {
        q = Q.size();
        ans.resize(q);
        double qtime = 0;
        FastBitset<uint64_t> vis(n);

        // 预 reserve：ef 上限 1700，给 2x 余量避免 push_heap 触发 realloc
        can_storage.reserve(4096);
        tp_storage.reserve(4096);

        for (auto &u : Q) {
            vis.reset();
            // 启动新查询：更新时间戳，使用缓存
            if (++dist_cur_ts == 0u) {
                // overflow: reset all timestamps to 0 and set curr to 1
                std::fill(dist_cache_ts.begin(), dist_cache_ts.end(), 0u);
                dist_cur_ts = 1u;
            }
            timeval t1, t2;
            gettimeofday(&t1, NULL);
            Qry(1, 0, n-1, u, vis, ans[u.id]);  // 直接写入 ans[u.id]，不再返回拷贝
            gettimeofday(&t2, NULL);
            qtime += GetTime(t1, t2);
        }
        return qtime;
    }

    double getIntersect(vector<int> u, vector<int> v) {
        if (u.empty()) return 0.0;
        set<int> s;
        for (auto e : u) s.insert(e);
        double cnt = 0;
        for (auto e : v) if (s.count(e)) cnt++;
        return cnt / static_cast<double>(u.size());
    }

    vector<vector<int>> Br;

    double getRecall() {
        double avg = 0;
        double d_err = 0, ratio = 0;
        int val_num = 0;
        for (int i = 0; i < q; i++) {
            vector<int> u;
            set<int> uu;
            for (int j : ans[i]) { uu.insert(j); ouf << j << " "; }
            ouf << endl;
            for (auto j : uu) u.push_back(j);
            double per = getIntersect(Br[i], u);
            double u_ratio = 0, d_ratio = 0;
            const int ku = std::min(K, (int)std::min(u.size(), Br[i].size()));
            auto query_data = (void*)(queryData.data() + Q[i].u * D);
            // Rank-aligned distance error (Patella et al.): i-th by ascending dist to q, not by id.
            // set<int> iteration order was by vertex id — that inflated d_err vs paper definition.
            std::sort(u.begin(), u.end(), [&](int a, int b) {
                float da = fstdistfunc_(query_data, getDataByInternalId(a), dist_func_param_);
                float db = fstdistfunc_(query_data, getDataByInternalId(b), dist_func_param_);
                return da < db || (da == db && a < b);
            });
            std::vector<int> br = Br[i];
            std::sort(br.begin(), br.end(), [&](int a, int b) {
                float da = fstdistfunc_(query_data, getDataByInternalId(a), dist_func_param_);
                float db = fstdistfunc_(query_data, getDataByInternalId(b), dist_func_param_);
                return da < db || (da == db && a < b);
            });
            if (const char *e_ef = std::getenv("METRIC_DUMP_EF")) {
                if (const char *e_q = std::getenv("METRIC_DUMP_QUERY")) {
                    if (ef == std::atoi(e_ef) && i == std::atoi(e_q)) {
                        std::cerr << "[FRASP_METRIC_DUMP] query=" << i << " ef=" << ef << " K=" << K << " ku=" << ku
                                  << "\n";
                        std::cerr << "  u_rank (dist asc): ";
                        for (int j = 0; j < ku; j++)
                            std::cerr << u[j] << (j + 1 == ku ? '\n' : ' ');
                        std::cerr << "  br_rank (dist asc): ";
                        for (int j = 0; j < ku; j++)
                            std::cerr << br[j] << (j + 1 == ku ? '\n' : ' ');
                        for (int j = 0; j < ku; j++) {
                            float dour =
                                fstdistfunc_(query_data, getDataByInternalId(u[j]), dist_func_param_);
                            float dgt =
                                fstdistfunc_(query_data, getDataByInternalId(br[j]), dist_func_param_);
                            std::cerr << "  j=" << j << " u=" << u[j] << " br=" << br[j] << " dour=" << dour
                                      << " dgt=" << dgt;
                            if (dgt > 1e-9f)
                                std::cerr << " term=" << (static_cast<double>(dour / dgt - 1.0f));
                            std::cerr << "\n";
                        }
                    }
                }
            }
            for (int j = 0; j < ku; j++) {
                float dour = fstdistfunc_(query_data, getDataByInternalId(u[j]), dist_func_param_);
                float dgt  = fstdistfunc_(query_data, getDataByInternalId(br[j]), dist_func_param_);
                if (dgt > 1e-9f) { ++val_num; d_err += static_cast<double>(dour / dgt - 1.0f); }
                u_ratio += static_cast<double>(dour);
                d_ratio += static_cast<double>(dgt);
            }
            if (d_ratio > 1e-9) ratio += u_ratio / d_ratio;
            avg += per;
        }
        cout << " d_err:" << (val_num > 0 ? d_err / val_num : 0.0) << " ratio: " << ratio/q
             << " dcnt: " << dcnt << " ";
        dcnt = 0;
        avg /= q;
        return avg;
    }

    void loadData(std::ifstream& in, std::vector<int>& data) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        data.clear(); data.resize(size);
        in.read(reinterpret_cast<char*>(data.data()), size * sizeof(int));
    }

    void loadData(std::ifstream& in, std::vector<std::vector<int>>& data) {
        size_t outerSize;
        in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
        data.clear(); data.resize(outerSize);
        for (auto& vec : data) {
            size_t innerSize;
            in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
            vec.resize(innerSize);
            in.read(reinterpret_cast<char*>(vec.data()), innerSize * sizeof(int));
        }
    }

    void loadData(std::ifstream& in,
                  std::vector<std::vector<std::vector<std::pair<int,int>>>>& data) {
        size_t outerSize;
        in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
        data.clear(); data.resize(outerSize);
        for (auto& matrix : data) {
            size_t midSize;
            in.read(reinterpret_cast<char*>(&midSize), sizeof(midSize));
            matrix.resize(midSize);
            for (auto& vec : matrix) {
                size_t innerSize;
                in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
                vec.resize(innerSize);
                in.read(reinterpret_cast<char*>(vec.data()),
                        innerSize * sizeof(std::pair<int,int>));
            }
        }
    }

    void loadData(std::ifstream& in,
                  std::vector<std::vector<std::vector<int>>>& data) {
        size_t outerSize;
        in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
        data.clear(); data.resize(outerSize);
        for (auto& matrix : data) {
            size_t midSize;
            in.read(reinterpret_cast<char*>(&midSize), sizeof(midSize));
            matrix.resize(midSize);
            for (auto& vec : matrix) {
                size_t innerSize;
                in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
                vec.resize(innerSize);
                in.read(reinterpret_cast<char*>(vec.data()), innerSize * sizeof(int));
            }
        }
    }

    float GetTime(timeval &begin, timeval &end) {
        return end.tv_sec - begin.tv_sec +
               (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC;
    }

    vector<vector<int>> run(const string& path, bool op, bool op2) {
        cerr << "N (Point): " << n << endl;
        cerr << "Q (Query): " << Q.size() << endl;
        cout << "build EF -> " << ef << endl;
        L.resize(4*n+1, nullptr);
        R.resize(4*n+1, nullptr);

        L2Space space(D);
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        data_size_ = (D + 7) / 8 * 8 * sizeof(float);
        prefetch_lines = data_size_ >> 4;
        sv(1, 0, n-1);

        vector<int> ND(4*n+1, 0);
        vector<int> Le(4*n+1, 0);
        vector<int> Re(4*n+1, 0);
        for (auto u : TN) ND[u.p] = 1, Le[u.p] = u.l, Re[u.p] = u.r;

        H = new HierarchicalNSW<float>(&space, n, M, ef);
        frasp_release_query_scaffold(H);
        std::ifstream input(path, std::ios::binary);

        loadData(input, H->st.a);
        frasp_skip_int_vector_blob(input);  // st.pr — rebuilt by reBuild()
        frasp_skip_int_vector_blob(input);  // st.f
        // On-disk ed[level][v] is already the HNSW adjacency list (see buildFRASP.h save).
        // Do not connect consecutive neighbors in that list; that was a mistaken "expansion"
        // that blew up degrees far beyond O(M).
        frasp_load_ed_as_csr(input, *H);
        frasp_load_del_as_csr(input, *H);
        H->st.mid = ((0 + n - 1) / 2);
        H->reBuild();

        for (size_t i = 0; i < L.size(); ++i) {
            int l = Le[i], r = Re[i], mid = (l + r) >> 1;
            if (ND[i]) {
                L[i] = new HierarchicalNSW<float>(&space, mid-l+1, M, ef);
                frasp_release_query_scaffold(L[i]);
                L[i]->st.mid = mid - ((l + mid) / 2 + 1);
                loadData(input, L[i]->st.a);
                frasp_skip_int_vector_blob(input);
                frasp_skip_int_vector_blob(input);
                frasp_load_ed_as_csr(input, *L[i]);
                frasp_load_del_as_csr(input, *L[i]);
                L[i]->reBuild();
            }
            if (ND[i]) {
                R[i] = new HierarchicalNSW<float>(&space, r-mid, M, ef);
                frasp_release_query_scaffold(R[i]);
                R[i]->st.mid = (r + mid + 1) / 2 - (mid + 1);
                loadData(input, R[i]->st.a);
                frasp_skip_int_vector_blob(input);
                frasp_skip_int_vector_blob(input);
                frasp_load_ed_as_csr(input, *R[i]);
                frasp_load_del_as_csr(input, *R[i]);
                R[i]->reBuild();
            }
        }
        input.close();
        puts("Finish Read Build");

        const bool assert_ed_caps = frasp_assert_ed_degree_caps();
        if (assert_ed_caps) {
            std::cerr << "FRASP_ASSERT_ED_DEGREES or FRASP_ASSERT_MAX_ED0_DEG=1: "
                         "checking all ed[][] vs maxM0_(layer0) and maxM_(upper)...\n";
        }

        double avd = 0, smd = 0;
        size_t mxdeg0 = frasp_check_hnsw_ed_degrees(H, "H", assert_ed_caps);
        for (int i = 0; i < n; i++) {
            smd++;
            auto [eb, ee] = H->ed_neighbors(0, i);
            avd += static_cast<double>(ee - eb);
        }
        cout << avd / smd << "av degre" << endl;
        cerr << "layer0 max_degree(H)=" << mxdeg0 << "  M=" << M
             << "  (HNSW caps: maxM0_=2*M=" << H->maxM0_ << ", maxM_=M=" << H->maxM_
             << ")\n";

        size_t mxdeg0_LR = 0;
        for (size_t i = 0; i < L.size(); ++i) {
            if (!ND[(int)i]) continue;
            char tagL[32], tagR[32];
            std::snprintf(tagL, sizeof tagL, "L[%zu]", i);
            std::snprintf(tagR, sizeof tagR, "R[%zu]", i);
            if (L[i])
                mxdeg0_LR = std::max(mxdeg0_LR,
                                     frasp_check_hnsw_ed_degrees(L[i], tagL, assert_ed_caps));
            if (R[i])
                mxdeg0_LR = std::max(mxdeg0_LR,
                                     frasp_check_hnsw_ed_degrees(R[i], tagR, assert_ed_caps));
        }
        cerr << "layer0 max_degree(among all L/R loaded)=" << mxdeg0_LR
             << "  (same caps per graph: maxM0_=" << (size_t)(2 * M) << " when M matches build)\n";

        if (assert_ed_caps)
            std::cerr << "FRASP ed degree assert passed (H + all L/R, all layers).\n";

        if (frasp_env_flag_on("FRASP_INDEX_EDGE_STATS"))
            frasp_print_index_edge_stats(H, L, R, ND);
        if (frasp_env_flag_on("FRASP_INDEX_EDGE_STATS_ONLY")) {
            for (size_t i = 0; i < L.size(); ++i) {
                delete L[i];
                delete R[i];
            }
            delete H;
            H = nullptr;
            puts("Finish");
            return {};
        }

        printMemoryUsage();
        // Initialize distance cache arrays sized by dataset
        dist_cache.assign(n, 0.0f);
        dist_cache_ts.assign(n, 0u);
        dist_cur_ts = 1u;
        std::vector<int> SearchEF;
        if (const char *s = std::getenv("FRASP_SINGLE_EF"); s && *s) {
            SearchEF = { std::atoi(s) };
            cerr << "FRASP_SINGLE_EF=" << SearchEF[0] << " (single-EF benchmark mode)\n";
        } else {
            SearchEF = {1700,1400,1100,1000,900,800,700,600,500,400,
                        300,250,200,180,160,140,120,100,90,80,
                        70,60,55,50,45,40,35,30,25,20,15,10};
        }
        for (auto o : SearchEF) {
            ef = o;
            cout << "(EF: " << o << "";
            queryTime = svQuery();
            cout << ", Query Time: " << queryTime << "s";
            cout << ", QueryPerSec: " << q / queryTime << "";
            double re = getRecall();
            cout << ", Recall: " << re;
            cout << ")" << endl;
        }
        printMemoryUsage();
        for (size_t i = 0; i < L.size(); ++i) {
            delete L[i];
            delete R[i];
        }
        return ans;
    }
};