#ifndef HPX_GRAPHFLOW_CORE_EXECUTORS_SIMPLE_EXECUTOR_HPP_
#define HPX_GRAPHFLOW_CORE_EXECUTORS_SIMPLE_EXECUTOR_HPP_

#include "core/tensor/tensor.hpp"
#include "core/operations/operation_context.hpp"
#include "core/graph/graph.hpp"
#include "executor.hpp"

#include <vector>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/unwrapped.hpp>

namespace graphflow { namespace executors {
    
class simple_executor : public executor {
public:
    hpx::future<void> run(
            graph::graph const& g,
            std::vector<std::pair<std::string, tensor> > const& feeds,
            std::vector<std::string> const& fetches
        ) override
    {        
        auto nodes = g.get_nodes();
        
        std::vector<hpx::future<void> > comp_futures;
        comp_futures.reserve(nodes.size());

        for (auto n : nodes)
        {                
            auto f = n->get_input_futures();
                                            
            comp_futures.push_back(
                hpx::when_all(f).then(
                    hpx::util::unwrapped2(
                        [n](std::vector<tensor> input)
                        {
                            auto op = n->get_operation();

                            operations::operation_context ctx;
                            
                            ctx.id = n->get_id();
                            
                            auto res = (*op)(ctx, input);
                            
                            n->set_output_promises(res);
                        }
                    )
                )
            );
        }
                                    
        return hpx::when_all(comp_futures);
    }
};
    
    
}} //namespace

#endif