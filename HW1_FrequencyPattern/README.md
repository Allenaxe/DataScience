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
---
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
* **minimum_support (Unsigned long long int):** Minimum support redefined by *minimun support rate * transaction size* 

#### Instance Function:
* **FPTREE(const vector<TRANSACTION>&, uint64_t):** FPTREE constructor
* **empty():** check whether a FPTREE is empty

### FPTREE Constructor
#### Function Variable
```c++
FPTREE::FPTREE(const vector<TRANSACTION>& transactions, uint64_t minimum_support) :
    root( make_shared<FPNODE>( ITEM{}, nullptr ) ), header_table(),
    minimum_support( minimum_support )
```
* **transactions:** Initiatialized by passing value.
* **minimum_support:** Initiatialized by passing value.
* **root:** Initiatialized by empty items and null pointer.
* **header_table:** Initiatialized by empty value.
#### Data Preprocessing
Counting the frequency of each items, for instance, **{0: 6, 1: 2}** means item 0 appears six times and item 1 appears two times in this dataset.
```c++
map<ITEM, uint64_t> frequency_ordered_by_item;
for ( const TRANSACTION& transaction : transactions ) {
  for ( const ITEM& item : transaction ) {
    ++frequency_ordered_by_item[item];
  }
}
```
* **frequency_ordered_by_item:** a map stores item-frequency pair sorted in increasing order by item index.
---
Check whether each items exceed minimim support. If not, remove it from dataset. By Downward Closure Property, if there any itemset which is infrequent, its superset should not be tested.
```c++
for ( auto it = frequency_ordered_by_item.cbegin(); it != frequency_ordered_by_item.cend(); ) {
  const uint64_t item_frequency = (*it).second;
    if ( item_frequency < minimum_support ) { frequency_ordered_by_item.erase( it++ ); }
    else { ++it; }
}
```
---
Order each items by its frequency by redefined the comparator of set.
```c++
struct frequency_comparator
{
  bool operator()(const pair<ITEM, uint64_t> &lhs, const pair<ITEM, uint64_t> &rhs) const
  {
      return tie(lhs.second, lhs.first) > tie(rhs.second, rhs.first);
  }
};

set<pair<ITEM, uint64_t>, frequency_comparator> items_ordered_by_frequency(frequency_ordered_by_item.cbegin(), frequency_ordered_by_item.cend());
```
* **items_ordered_by_frequency:** a set stores item-frequency pair sorted in decreasing order by frequency.
---
#### Construct FPTREE
Construct FPTREE by the order of transactions and grow up the tree from root by items.

1. For each transactions, iterate items by its frequency and find its place in transaction. That is, extract the most frequent item and delete it from each transaction.
2. Find the item in the children of current fpnode.
3. If it exists, increase the frequnecy of the certain chlid node and move current fpnode to it.
4. If it doesn't exist, construct new child node, connect it to current fpnode, and update header table.
5. If the item exists in header table, iterate the node link path and append it into the end (header table is like a table of link list. That is, key is item index and value is the head pointer of link list).
6. If the item doesn't exists in header table, initiate key-value pair {item: new child node}.
```c++
for ( const TRANSACTION& transaction : transactions ) {
  auto current_fpnode = root;

  for ( const auto& pair : items_ordered_by_frequency ) {
    const ITEM& item = pair.first;

    if ( find( transaction.cbegin(), transaction.cend(), item ) != transaction.cend() ) {

      const auto it = find_if(
        current_fpnode->children.cbegin(), current_fpnode->children.cend(),  [&](const shared_ptr<FPNODE>& fpnode) {
          return fpnode->item == item;
      } );

      if ( it == current_fpnode->children.cend() ) {
        const auto current_fpnode_new_child = make_shared<FPNODE>( item, current_fpnode );

        current_fpnode->children.push_back( current_fpnode_new_child );

        if ( header_table.count( current_fpnode_new_child->item ) ) {
          auto prev_fpnode = header_table[current_fpnode_new_child->item];
          while ( prev_fpnode->node_link ) { prev_fpnode = prev_fpnode->node_link; }
          prev_fpnode->node_link = current_fpnode_new_child;
        }
        else {
          header_table[current_fpnode_new_child->item] = current_fpnode_new_child;
        }

        current_fpnode = current_fpnode_new_child;
      }
      else {
        auto current_fpnode_child = *it;
        ++current_fpnode_child->frequency;
        current_fpnode = current_fpnode_child;
      }
    }
  }
}
```

> [!NOTE]  
> **Smart pointer (C++ 11 stardard):**  
> * **shared_ptr (Strong reference pointer):** Allowed mutiple shared_ptr points to the same resoure, and this action will increase value of the counter in the shared_ptr. All shared_ptr is released, the resource is released.  
> * **weak_ptr (Weak reference pointer):** It supports shred_ptr. It can only access the resource and know whether it survives, but it cannnot control the lifetime of resource. 
