#include <iostream>
#include <algorithm>
#include <fstream>
#include <utility>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include <set>
#include <map>

using namespace std;

using ITEM = uint64_t;
using TRANSACTION = vector<ITEM>;
using PREFIXPATH = pair<vector<ITEM>, uint64_t>;
using PATTERN = pair<set<ITEM>, uint64_t>;

struct FPNODE {
    const ITEM item;
    uint64_t frequency;
    shared_ptr<FPNODE> node_link;
    weak_ptr<FPNODE> parent;
    vector<shared_ptr<FPNODE>> children;

    FPNODE(const ITEM&, const shared_ptr<FPNODE>&);
};

struct FPTREE {
    shared_ptr<FPNODE> root;
    map<ITEM, shared_ptr<FPNODE>> header_table;
    uint64_t minimum_support;

    FPTREE(const vector<TRANSACTION>&, uint64_t);

    bool empty() const;
};

struct item_comparator
{
    bool operator() (const PATTERN& lhs, const PATTERN& rhs) const
    {
        if(lhs.first.size() != rhs.first.size()) return lhs.first.size() < rhs.first.size();
        else return lhs < rhs;
    }
};

FPNODE::FPNODE(const ITEM& item, const shared_ptr<FPNODE>& parent) :
    item( item ), frequency( 1 ), node_link( nullptr ), parent( parent ), children() {}

FPTREE::FPTREE(const vector<TRANSACTION>& transactions, uint64_t minimum_support) :
    root( make_shared<FPNODE>( ITEM{}, nullptr ) ), header_table(),
    minimum_support( minimum_support )
{
    map<ITEM, uint64_t> frequency_ordered_by_item;
    for ( const TRANSACTION& transaction : transactions ) {
        for ( const ITEM& item : transaction ) {
            ++frequency_ordered_by_item[item];
        }
    }

    for ( auto it = frequency_ordered_by_item.cbegin(); it != frequency_ordered_by_item.cend(); ) {
        const uint64_t item_frequency = (*it).second;
        if ( item_frequency < minimum_support ) { frequency_ordered_by_item.erase( it++ ); }
        else { ++it; }
    }

    struct frequency_comparator
    {
        bool operator()(const pair<ITEM, uint64_t> &lhs, const pair<ITEM, uint64_t> &rhs) const
        {
            return tie(lhs.second, lhs.first) > tie(rhs.second, rhs.first);
        }
    };

    set<pair<ITEM, uint64_t>, frequency_comparator> items_ordered_by_frequency(frequency_ordered_by_item.cbegin(), frequency_ordered_by_item.cend());

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
}

bool FPTREE::empty() const
{
    return root->children.size() == 0;
}


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

int main( int argc, char *argv[] )
{

    //auto start = std::chrono::system_clock::now();

    cin.tie(nullptr); cout.tie(nullptr); ios_base::sync_with_stdio(false);

    double minimum_support_rate = atof( argv[1] );
    ifstream in( argv[2] );
    string input;
    vector <TRANSACTION> transactions;
    while ( getline( in, input ) ) {
        TRANSACTION transaction;
        stringstream ss( input );
        char dilameter = ',';
        while ( getline(ss, input, dilameter) ) {
            transaction.push_back(stoll(input));
        }
        transactions.push_back(transaction);
    }

    const uint64_t minimum_support = ceil( minimum_support_rate * transactions.size() );

    const FPTREE fptree{ transactions, minimum_support };

    const set<PATTERN, item_comparator> patterns = fptree_growth( fptree );

    in.close();

    ofstream out( argv[3] );

    for ( auto pattern : patterns ) {
        string output;
        for( auto item : pattern.first ) {
            output += to_string( item ) + ',';
        }
        output.back() = ':';
        out << output;
        out << fixed << setprecision(4) << (double)pattern.second / 20.0 << '\n';
    }

    out.close();

    /*  timestamp
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    cout << "-------------------------------------------" << '\n';
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << '\n';
    */
    
}