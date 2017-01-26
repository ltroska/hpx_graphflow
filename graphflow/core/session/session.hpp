#ifndef HPX_GRAPHFLOW_CORE_SESSION_SESSION_HPP_
#define HPX_GRAPHFLOW_CORE_SESSION_SESSION_HPP_

#include "core/graph/graph.hpp"
#include "core/executors/simple_executor.hpp"

namespace graphflow {

class session
{
public:
    session() {}

    void run(graph::graph& g,
        std::vector<std::pair<std::string, tensor> > const& feeds,
        std::vector<std::string> const& fetches,
        std::vector<tensor>& outputs)
    {        
        std::unique_ptr<executors::executor> exec =
            std::make_unique<executors::simple_executor>();
            
        g.setup_dependencies();
            
        outputs = exec->run(g, feeds, fetches);
    }
};


}

#endif