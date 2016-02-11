#include "ordering_container.h"

#include <iostream>
using std::cout;
using std::endl;

OrderingContainer::OrderingContainer()
{
}

void OrderingContainer::clear()
{
    id_to_priority_.clear();
    priority_to_id_.clear();
}

void OrderingContainer::prepareIDAndDeletePriorPriority(int id)
{
    if (id >= id_to_priority_.size()) {
        id_to_priority_.resize(id+1, -1);
    }
    else {
        // remove any existing priority
        int prior_priority = id_to_priority_[id];
        if (prior_priority >= 0) {
            std::map<int,int>::iterator find_iter = priority_to_id_.find(prior_priority);
            if (find_iter == priority_to_id_.end()) {
                cout << "OrderingContainer" << endl;
                throw std::runtime_error("OrderingContainer");
            }
            if (find_iter->second != id) {
                cout << "OrderingContainer" << endl;
                throw std::runtime_error("OrderingContainer");
            }
            priority_to_id_.erase(find_iter);
        }
    }
}

void OrderingContainer::setIDToMin(int id)
{
    prepareIDAndDeletePriorPriority(id);

    int new_priority = 0;
    if (!priority_to_id_.empty()) {
        new_priority = priority_to_id_.begin()->first - 1;
    }
    id_to_priority_[id] = new_priority;
    priority_to_id_.insert(std::make_pair(new_priority,id));
}

void OrderingContainer::setIDToMax(int id)
{
    prepareIDAndDeletePriorPriority(id);

    int new_priority = 0;
    if (!priority_to_id_.empty()) {
        new_priority = priority_to_id_.rbegin()->first + 1;
    }
    id_to_priority_[id] = new_priority;
    priority_to_id_.insert(std::make_pair(new_priority,id));
}

void OrderingContainer::getIterators(std::map<int,int>::const_iterator & result_begin, std::map<int,int>::const_iterator & result_end)
{
    result_begin = priority_to_id_.begin();
    result_end = priority_to_id_.end();
}
