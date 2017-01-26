#ifndef HPX_GRAPHFLOW_CORE_EXECUTORS_EXECUTOR_HPP_
#define HPX_GRAPHFLOW_CORE_EXECUTORS_EXECUTOR_HPP_

#include "core/tensor/tensor.hpp"
#include "core/graph/graph.hpp"

#include <vector>

namespace graphflow { namespace executors {
    
class executor {
    public:
        virtual std::vector<tensor> run(graph::graph const& g,
            std::vector<std::pair<std::string, tensor> > const& feeds,
            std::vector<std::string> const& fetches) = 0;
};
    
    
}} //namespace

#endif