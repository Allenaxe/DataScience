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
#### FP-Growth

Mining frequent itemset through header table from bottom to the top. To every items in header table corresponding to FPTREE, find its conditional pattern base. Conditional pattern base means the subtree which the mining node is the leaf node. Set the counters of every nodes to be the value of leaf node and delete the nodes which its value is lower the minimum support. Based on this conditional pattern base, mine the frequent itemset recursively.

##### Utility

Check whether a path is a single path which is each nodes in this path contain one child excluding leaf node.

```c++
bool single_path(const shared_ptr<FPNODE>& fpnode)
{
  if ( fpnode->children.size() == 0 ) { return true; }
  if ( fpnode->children.size() > 1 ) { return false; }
  return single_path( fpnode->children.front() );
}

bool single_path(const FPTREE& fptree)
{
  return fptree.empty() || single_path( fptree.root );
}
```
---
1. For the FPTREE with single path, iteratively dig into it and add item to the set to become a new pattern, and the insert it into single_path to become a part of answer.

```c++
set<PATTERN, item_comparator> fptree_growth(const FPTREE& fptree)
{
  if ( fptree.empty() ) { return {}; }

  if ( single_path( fptree ) ) {

    set<PATTERN, item_comparator> single_path;

    auto current_fpnode = fptree.root->children.front();
    while ( current_fpnode ) {
      const ITEM& current_fpnode_item = current_fpnode->item;
      const uint64_t current_fpnode_frequency = current_fpnode->frequency;

      PATTERN new_pattern{ { current_fpnode_item }, current_fpnode_frequency };
      single_path.insert( new_pattern );

      for ( const PATTERN& pattern : single_path ) {
        PATTERN new_pattern{ pattern };

        new_pattern.first.insert( current_fpnode_item );
        new_pattern.second = current_fpnode_frequency;

        single_path.insert( new_pattern );
      }

      if ( current_fpnode->children.size() == 1 ) { current_fpnode = current_fpnode->children.front(); }
      else { current_fpnode = nullptr; }
    }

    return single_path;
  }
  else {

    set<PATTERN, item_comparator> multi_path;

    for ( const auto& pair : fptree.header_table ) {
      const ITEM& current_item = pair.first;

      vector<PREFIXPATH> conditional_pattern_base;

      auto starting_fpnode = pair.second;
      while ( starting_fpnode ) {
        const uint64_t starting_fpnode_frequency = starting_fpnode->frequency;

        auto current_path = starting_fpnode->parent.lock();
        if ( current_path->parent.lock() ) {
          PREFIXPATH prefix_path{ {}, starting_fpnode_frequency };

          while ( current_path->parent.lock() ) {
              prefix_path.first.push_back( current_path->item );

              current_path = current_path->parent.lock();
          }

          conditional_pattern_base.push_back( prefix_path );
        }

        starting_fpnode = starting_fpnode->node_link;
      }
      vector<TRANSACTION> conditional_fptree_transactions;
      for ( const PREFIXPATH& prefix_path : conditional_pattern_base ) {
        const vector<ITEM>& prefix_path_items = prefix_path.first;
        const uint64_t prefix_path_items_frequency = prefix_path.second;

        TRANSACTION transaction = prefix_path_items;

        for ( int i = 0; i < prefix_path_items_frequency; ++i ) {
          conditional_fptree_transactions.push_back( transaction );
        }
      }

      const FPTREE conditional_fptree( conditional_fptree_transactions, fptree.minimum_support );
      set<PATTERN, item_comparator> conditional_patterns = fptree_growth( conditional_fptree );

      set<PATTERN, item_comparator> current_item_patterns;

      uint64_t current_item_frequency = 0;
      auto fpnode = pair.second;
      while ( fpnode ) {
        current_item_frequency += fpnode->frequency;
        fpnode = fpnode->node_link;
      }
      PATTERN pattern{ { current_item }, current_item_frequency };
      current_item_patterns.insert( pattern );

      for ( const PATTERN& pattern : conditional_patterns ) {
        PATTERN new_pattern{ pattern };
        new_pattern.first.insert( current_item );
        new_pattern.second = pattern.second;

        current_item_patterns.insert( { new_pattern } );
      }

      multi_path.insert( current_item_patterns.cbegin(), current_item_patterns.cend() );
    }

    return multi_path;
  }
}
```

> [!NOTE]  
> **Smart pointer (C++ 11 stardard):**  
> * **shared_ptr (Strong reference pointer):** Allowed mutiple shared_ptr points to the same resoure, and this action will increase value of the counter in the shared_ptr. All shared_ptr is released, the resource is released.  
> * **weak_ptr (Weak reference pointer):** It supports shared_ptr. It can only access the resource and know whether it survives, but it cannnot control the lifetime of resource. 
