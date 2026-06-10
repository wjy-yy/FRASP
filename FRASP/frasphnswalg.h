#pragma once

#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdint>

struct ST{
    int n, mid;
    std::vector<int> a, pr, f;
    int max(int x, int y) const {
            return a[x] > a[y] ? x : y;
    }
    void clr() {
        a.clear();
        pr.clear();
        f.clear();
    }
    void inline STPrework() {
        pr.resize(n, 0);
        f.resize(n, 0);
        for (int i = 0; i < n; i++) {
            pr[i] = i;
            if (i) pr[i] = max(pr[i], pr[i - 1]);
        }

        f[mid] = mid, f[mid + 1] = mid + 1;
        for (int i = mid + 2; i < n; i++) {
            f[i] = max(i, f[i - 1]);
        }

        for (int i = mid - 1; i >= 0; i--) {
            f[i] = max(i, f[i + 1]);
        }
   
    }

    int inline query(int l, int r) const  {
        if (l > r) std::swap(l, r);
        if (l <= mid && mid < r) {
            return max(f[l], f[r]);
        } else if (l == 0) {
            return pr[r];
        } else {
            std::cout<<l<<" "<<r<<std::endl;
            assert(false);
        }
    }
   
} ;

namespace hnswlib {


typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0}; 
    
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };
    int tm {0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};
    tableint enterpoint_node_{0};
    std::vector<int> element_levels_; 
    

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};


    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};
    std::vector< std::vector<std::vector<int > > > ed;
    std::vector< std::vector<std::vector<int > > > cr;
    std::vector< std::vector<std::vector<std::pair<int, int> > > > del;
    // Query-only compact ed/del (CSR). Populated by query loader; build/index unchanged.
    std::vector<std::vector<uint32_t>> ed_csr_off_;
    std::vector<std::vector<int>> ed_csr_edges_;
    std::vector<std::vector<uint32_t>> del_csr_off_;
    std::vector<std::vector<std::pair<int, int>>> del_csr_edges_;
    std::vector<char* > dt;
    ST st;
    bool allow_replace_deleted_ = false;

    mutable std::vector<int> vis;
    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : 
            element_levels_(max_elements) {
        max_elements_ = max_elements;
       
       
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);
 
        cur_element_count = 0;
        vis.clear();
        vis.resize(max_elements, 0);
        enterpoint_node_ = -1;
        maxlevel_ = -1;
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        cur_element_count = 0;
        
        vis.clear();
        ed.clear();
        ed_csr_off_.clear();
        ed_csr_edges_.clear();
        st.clr();
        del.clear();
        del_csr_off_.clear();
        del_csr_edges_.clear();
        dt.clear();
    }

    bool ed_use_csr() const { return !ed_csr_off_.empty(); }

    inline std::pair<const int *, const int *> ed_neighbors(int level, int u) const {
        if (!ed_use_csr() || level < 0 || level >= (int)ed_csr_off_.size())
            return {nullptr, nullptr};
        const auto &off = ed_csr_off_[level];
        if (u < 0 || u + 1 >= (int)off.size())
            return {nullptr, nullptr};
        const auto &pool = ed_csr_edges_[level];
        return {pool.data() + off[u], pool.data() + off[u + 1]};
    }

    inline int ed_num_levels() const {
        if (ed_use_csr()) return (int)ed_csr_off_.size();
        return (int)ed.size();
    }

    inline int ed_num_vertices(int level) const {
        if (ed_use_csr()) {
            if (level < 0 || level >= (int)ed_csr_off_.size()) return 0;
            return (int)ed_csr_off_[level].size() - 1;
        }
        if (level < 0 || level >= (int)ed.size()) return 0;
        return (int)ed[level].size();
    }

    bool del_use_csr() const { return !del_csr_off_.empty(); }

    inline std::pair<const std::pair<int, int> *, const std::pair<int, int> *>
    del_neighbors(int level, int u) const {
        if (!del_use_csr() || level < 0 || level >= (int)del_csr_off_.size())
            return {nullptr, nullptr};
        const auto &off = del_csr_off_[level];
        if (u < 0 || u + 1 >= (int)off.size())
            return {nullptr, nullptr};
        const auto &pool = del_csr_edges_[level];
        return {pool.data() + off[u], pool.data() + off[u + 1]};
    }

    inline int del_num_levels() const {
        if (del_use_csr()) return (int)del_csr_off_.size();
        return (int)del.size();
    }

    inline int del_num_vertices(int level) const {
        if (del_use_csr()) {
            if (level < 0 || level >= (int)del_csr_off_.size()) return 0;
            return (int)del_csr_off_[level].size() - 1;
        }
        if (level < 0 || level >= (int)del.size()) return 0;
        return (int)del[level].size();
    }




    void setEf(size_t ef) {
        ef_ = ef;
    }



    inline labeltype getExternalLabel(tableint internal_id) const {
        return internal_id;
    }
    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    inline char *getDataByInternalId(tableint internal_id) const {
        return dt[internal_id];
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

 

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
       
        int visited_array_tag = 1;
        std::vector<int> z;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
        
        dist_t lowerBound;
         if (true) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        vis[ep_id] = visited_array_tag;
        z.push_back(ep_id);
        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            int u = curNodeNum;
            if (layer < ed.size() && u < ed[layer].size()) 
            for (int candidate_id: ed[layer][u]) {
                if (candidate_id == 0) continue;
                if (vis[candidate_id] == visited_array_tag) continue;
                vis[candidate_id] = visited_array_tag;
                z.push_back(candidate_id);
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        
        for (int v: z) vis[v] = 0;
        return top_candidates;
    }


    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        int visited_array_tag = 1;
        std::vector<int> z;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        vis[ep_id] = visited_array_tag;
        z.push_back(ep_id);
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;

            int u = current_node_id;
            if (ed.size() && u < ed[0].size()) for (int candidate_id: ed[0][u]) {
                if (candidate_id == 0) continue;
                if (!(vis[candidate_id] == visited_array_tag)) {
                    vis[candidate_id] = visited_array_tag;
                    z.push_back(candidate_id);

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);

                        if (bare_bone_search || 
                            (((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        for (int v: z) vis[v] = 0;
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }



    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }
        

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            
            
            while (ed.size() <= level) {
                ed.push_back({});
            }
            while (ed[level].size() <= cur_c) {
                ed[level].push_back({});
            }
            for (int v: selectedNeighbors) {
                ed[level][cur_c].push_back(v);
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
           
            
            int o = selectedNeighbors[idx];
            bool is_cur_c_present = false;
            if (isUpdate) {
                
                if (level < ed.size() && o < ed[level].size()) for (int v: ed[level][o]) {
                    if (v == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            while (ed.size() <= level) {
                ed.push_back({});
            }
            while (ed[level].size() <= o) {
                ed[level].push_back({});
            }
            int sz_link_list_other = ed[level][o].size();
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    ed[level][o].push_back(cur_c);
                } else {
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);
                    for (int v: ed[level][o]) {
                        candidates.emplace(
                            fstdistfunc_(getDataByInternalId(v), getDataByInternalId(selectedNeighbors[idx]),
                                            dist_func_param_), v);
                    }
                    getNeighborsByHeuristic2(candidates, Mcurmax);

            
                    while (del.size() <= level) {
                        del.push_back({});
                    }
                    while (del[level].size() < max_elements_) {
                        del[level].push_back({});
                    }
                    std::set<int> A;
                    std::set<int> B;
                    for (int v: ed[level][o]) {
                        A.insert(v);
                    }
                    int indx = 0;
                    while (candidates.size() > 0) {
                        int ne = ed[level][o][indx] = candidates.top().second;
                        candidates.pop();
                        B.insert(ne);
                        indx++;
                    }
                    ed[level][o].resize(indx);
                    int u = selectedNeighbors[idx];
                    for (int v: A) {
                        if (!B.count(v)) {
                             del[level][u].push_back({ v, tm });
                        }
                    }
                }
            }
        }

        return next_closest_entry_point;
    }



    size_t indexFileSize() const {
        size_t size = 0;
    
        size += sizeof(ST);
        size += st.a.capacity() * sizeof(int);
        size += st.pr.capacity() * sizeof(int);
        size += st.f.capacity() * sizeof(int);
    
        int sdbe = 0;
        for (const auto& layer : ed) {
            size += sizeof(layer);
            sdbe += layer.size() * sizeof(std::vector<int>);
            size += layer.capacity() * sizeof(std::vector<int>);
            for (const auto& node_edges : layer) {
                size += node_edges.capacity() * sizeof(int);
                sdbe += node_edges.size() * sizeof(int);
            }
        }
    
        return size;
    }

    void saveIndex(const std::string &location){}
 


    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

 
        bool is_vacant_place = 0;
        if (!is_vacant_place) {
            tm = label;
            addPoint(data_point, label, -1);
            
        } else {
     
        }
    }





    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
          

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
        }

       
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;
       
        int maxlevelcopy = maxlevel_;
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        dt.push_back((char*)data_point);
        

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                       
                        if (level < ed.size() && currObj < ed[level].size()) for (int cand: ed[level][currObj]) {
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

           bool epDeleted = 0;
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0) 
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    
     void reBuild() {
        int n = max_elements_;
        st.n = n;
        st.STPrework();
    }


    void prework() {
        int n = max_elements_;
        st.n = n;
        st.a.clear();
        for (int i = 0; i < n; i++) {
            st.a.push_back(element_levels_[i]);
        }
        for (int j = 0; j < del.size(); j++)
            for (int i = 0; i < del[j].size(); i++) {
                std::sort(del[j][i].begin(), del[j][i].end());
            }
        reBuild();
    }
    
     void buildST() {
        int n = max_elements_;
        st.n = n;
        st.a.clear();
        for (int i = 0; i < n; i++) {
            st.a.push_back(element_levels_[i]);
        }
        st.STPrework();
    }


    
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                if (level < ed.size() && currObj < ed[level].size())
                for (int cand: ed[level][currObj]) {
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


};
}
