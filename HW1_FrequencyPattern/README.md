# FP-Growth

## Basic Primitives
```c++
using ITEM = uint64_t;
using TRANSACTION = vector<ITEM>;
using PREFIXPATH = pair<vector<ITEM>, uint64_t>;
using PATTERN = pair<set<ITEM>, uint64_t>;
```
* **ITEM (Unsigned long long int):** Store item indexed by 0, 1, 2, ...
* **TRANSACTION (ITEM):** Store serveral item.
* **PREFIXPATH (Pair of Transaction and ):**
* **PATTERN (Pair of ITEM set and ):**
```c++
struct FPNODE {
    const ITEM item;
    uint64_t frequency;
    shared_ptr<FPNODE> node_link;
    weak_ptr<FPNODE> parent;
    vector<shared_ptr<FPNODE>> children;

    FPNODE(const ITEM&, const shared_ptr<FPNODE>&);
};
```

> [!NOTE]  
> Highlights information that users should take into account, even when skimming.