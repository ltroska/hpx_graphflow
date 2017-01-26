#ifndef HPX_GRAPHFLOW_CORE_SESSION_SESSION_HPP_
#define HPX_GRAPHFLOW_CORE_SESSION_SESSION_HPP_

#include "core/graph/graph.hpp"
#include "core/executors/simple_executor.hpp"

namespace graphflow {

class session
{
public:
    session() {}

    void run(
            graph::graph& g,
            std::vector<std::pair<std::string, tensor> > const& feeds,
            std::vector<std::string> const& fetches,
            std::vector<tensor>& outputs
        )
    {        
        std::unique_ptr<executors::executor> exec =
            std::make_unique<executors::simple_executor>();
            
        g.setup_dependencies();
        
        auto named_ops = g.get_named_ops();
        
        auto root = g.get_root();
                               
        for (auto const& p : feeds)
        {
            auto n = named_ops[p.first];
            
            n->set_input_future_from_feed(p.second);
        }
            
        auto exec_future = exec->run(g, feeds, fetches);
        
        exec_future.wait();
        
        outputs.reserve(fetches.size());
        
        for (auto const& p : fetches)
        {
            auto n = named_ops[p];
            
            n->get_output_future(0).then(
                hpx::util::unwrapped(
                    [&outputs](tensor t)
                    {
                        outputs.push_back(t);
                    }
                )
            );
        }
    }
};


}

#endif