# FRASP_review

FRASP is a high-performance index for range-filtering approximate nearest neighbor search, using a first-split segment tree strategy with suffix/prefix HNSW graphs to answer queries efficiently with three index lookups.

### Start to Compile

```sh
cd FRASP && mkdir build && cd build
cmake ..
make
```

### Construct Index

Command for using `./build`:

```sh
./build --data_path <path_to_base_data.fvecs> --index_file <path_to_save_index.bin> \
        --M 64 --ef 400 --B 1024 --threads 64
```

**parameters:**

<!-- `--data_prefix`: The path of folder where the base data, query data, query range and ground truth files saved. -->

- `--data_path`: The path of the base data to be processed. The data should be an `.fvecs` file, whose format is introduced in [fvecs datasets description](http://corpus-texmex.irisa.fr).
- `--index_file`: The constructed index will be saved to this file, in `.bin` format.
- `--M`: The degree of the graph index.
- `--ef`: The size of result set during index building.
- `--B` / `--K`: Segment-tree leaf threshold (must match at query time). Typical: **1024** for 1M datasets; **16384** (2^14) for Deep10M mixed workload.
- `--threads`: The number of threads for index building.

On completion, `<index_file>.build.log` records total build time, peak RSS, and index size.


### Query search

Command for using `./query`:

```sh
./query --data_path <path_to_base_data.fvecs>\
        --query_path <path_to_query_data.fvecs>\
        --index <path_to_index.bin>\
        --range_path <path_to_query_range>\ 
        --gt_path <path_to_groundtruth.fvecs>\ 
        --M 64 --ef 400 --B 1024
```

**parameters:**

- `--data_path`: The path to the base dataset (in `.fvecs` format), which has been used to construct the index.
- `--query_path`: The path to the query data (also in `.fvecs` format).
- `--index`: Path to the previously saved index file (in `.bin` format).
- `--range_path`: Path to the `.txt` file containing the range filters for each query.
- `--gt_path`: Path to the groundtruth file (in `.txt` format), where each line contains the top-`K` groundtruth neighbors for the corresponding query.
- `--M`: The degree of the graph index (must match the value used in construction).
- `--ef`: EF used at query time (search sweep; build used `--ef` during construction).
- `--B` / `--K`: Segment-tree leaf threshold (**must match** the `--B` used when building the index).
- `--threads`: OpenMP threads for query (optional, default 32).
- `--result_k`: Top-K for recall (optional, default 10).

#### Data format in `.fvecs` file

The `.fvecs` file format is a binary format used to store a large collection of float vectors compactly. It is widely adopted in vector search benchmarks (e.g., [Sift](http://corpus-texmex.irisa.fr)) for high-performance I/O and efficient storage.

Each vector in an `.fvecs` file is stored as:

`[int32: d][float32 x d: vector content]`

Where:

- `d`: A 32-bit signed integer indicating the dimension of the vector.
- `vector content`: A sequence of `d` floating point values (each 32-bit float, i.e., `float32`) representing the vector itself.

This pattern repeats for every vector stored in the file. Therefore, an `.fvecs` file containing `N` vectors of dimension `d` will have `N` blocks, each comprising:

4 bytes for the dimension `d`
followed by 4 * `d` bytes for the vector values

#### Explanation on range file

The range file is stored as a `.txt` file.

Each `.txt` file contains `N` lines. Each line represents a query range `[l,r]`, denoting the range filter.

### Instructions on running over Sift data

We have uploaded the Sift dataset to folder `sift_data`, which complies with GitHub's 100MB storage limit, while the other datasets are beyond the limit. The data sources of all datasets are listed at the end of README. 

To run on Sift dataset, the data file needs to be decompressed by using `tar -jxvf sift.tar.bz <file_directory>`.

`sift_base.fvecs` is the base data (1000000 objects). `sift_query.fvecs` is the query data (1000 queries). `range.txt` contains 1000 lines, each line of 2 integers, denoting 1000 range of 2^-2 fraction workload corresponding to each query. `gt.txt` contains 1000 lines, each line of K integers (K=10), denoting the groundtruth top K nearest neighbors from top-1 to top-`K` corresponding to each query. 

After decompressing the data, the following command can be used.

```sh
# decompress the data
cd FRASP/sift_data && tar -jxvf sift.tar.bz

# go to the build directory
cd FRASP/build && ./build --data_path ../sift_data/sift_base.fvecs --index_file index.bin --M 64 --ef 400 --B 1024 --threads 64

# query after building
./query --data_path ../sift_data/sift_base.fvecs \
        --query_path ../sift_data/sift_query.fvecs \
        --index index.bin \
        --range_path ../sift_data/range.txt \
        --gt_path ../sift_data/gt.txt \
        --M 64 --ef 400 --B 1024
```

### Datasets
The links of the datasets used in experiments are listed as follows:
- Sift: http://corpus-texmex.irisa.fr/
- Gist: http://corpus-texmex.irisa.fr/
- YT-rgb: https://research.google.com/youtube8m/
- WIT: https://github.com/google-research-datasets/wit
- Deep10M: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search

### Baseline Codes

The baseline codes are available at:

- **WoW**: [https://github.com/nju-websoft/WoW](https://github.com/nju-websoft/WoW) (benchmark: [https://github.com/ziqiwww/wow_benchmark](https://github.com/ziqiwww/wow_benchmark))
- **UNIFY**: [https://github.com/sjtu-dbgroup/UNIFY](https://github.com/sjtu-dbgroup/UNIFY)
- iRangeGraph: [https://github.com/YuexuanXu7/iRangeGraph](https://github.com/YuexuanXu7/iRangeGraph)
- SeRF: [https://github.com/rutgers-db/SeRF](https://github.com/rutgers-db/SeRF)
- SuperPostFiltering: [https://github.com/JoshEngels/RangeFilteredANN](https://github.com/JoshEngels/RangeFilteredANN)
- ACORN: [https://github.com/guestrin-lab/ACORN](https://github.com/guestrin-lab/ACORN)
- Milvus: [https://github.com/milvus-io/milvus](https://github.com/milvus-io/milvus)

## Baseline Parameters

| Method | Parameters |
|---|---|
| **iRangeGraph** | WIT, YT-RGB: M=64; SIFT, GIST: M=16 (grid search); Deep10M: M=16 |
| **SuperPostFiltering** | M=64, ef_con=500, β=2 |
| **SeRF** | 2DSegmentGraph + MaxLeap, M=32, K=100 |
| **WoW** | o=4, m=16, ω_c=128 (SIFT), ω_c=256 (others) |
| **Milvus** | SIFT, Deep10M: M=16; WIT: M=64; others: M=32 |
| **ACORN** | M=32, γ=12, M_β=64 |
| **UNIFY (HSIG)** | S=8, M=16, m=16, ef_con=500, τ_A=0.01n, τ_B=0.5n |
