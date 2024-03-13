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

### FPNODE
The data structure of a node in fptree.
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
#### Instance Variable:
* **ITEM (ITEM):** Store item indexed by 0, 1, 2...
* **frequency (Unsigned long long int):** Store the frequecy counter for a node
* **node_link (FPNODE\*):**
* **parent (FPNODE\*):** Point to the parent node
* **children (vector of FPNODE\*):** Point to the children

#### Instance Function:
* **FPNODE(const ITEM&, const shared_ptr<FPNODE>&):** FPNODE constructor

### FPTREE
The data structure of a fptree.
```c++
struct FPTREE {
    shared_ptr<FPNODE> root;
    map<ITEM, shared_ptr<FPNODE>> header_table;
    uint64_t minimum_support;

    FPTREE(const vector<TRANSACTION>&, uint64_t);

    bool empty() const;
};
```
#### Instance Variable:
* **root (FPNODE\*):** Root of a FPTREE
* **header_table (map):** Header table
* **minimum_support: (Unsigned long long int):** Minimum support redefined by *minimun support rate * transaction size* 

#### Instance Function:
* **FPTREE(const vector<TRANSACTION>&, uint64_t):** FPTREE constructor
* **empty()** check whether a FPTREE is empty

> [!NOTE]  
> **Smart pointer (C++ 11 stardard):**  
> * shared_ptr (Strong reference pointer): Allowed mutiple shared_ptr points to the same resoure, and this action will increase value of the counter in the shared_ptr. All shared_ptr is released, the resource is released.  
> * weak_ptr (Weak reference pointer): It Supports shred_ptr. It can only access the resource and know whether it survives, but it cannnot control the lifetime of resource. 
