#pragma once

#include <vector>
#include <map>

class OrderingContainer
{
public:
    OrderingContainer();

    // make empty
    void clear();

    // to look at id first
    void setIDToMin(int id);

    // to look at id last
    void setIDToMax(int id);

    void getIterators(std::map<int,int>::const_iterator & result_begin, std::map<int,int>::const_iterator & result_end);
    

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
		ar & id_to_priority_;
		ar & priority_to_id_;
	}

protected:

    void prepareIDAndDeletePriorPriority(int id);


    std::vector<int> id_to_priority_;
    std::map<int,int> priority_to_id_;

};