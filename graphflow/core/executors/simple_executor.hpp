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
    std::vector<tensor> run(graph::graph const& g,
        std::vector<std::pair<std::string, tensor> > const& feeds,
        std::vector<std::string> const& fetches)
        override
    {            
        auto nodes = g.get_nodes();
        auto named_ops = g.get_named_ops();
        
        auto root = g.get_root();
                               
        for (auto const& p : feeds)
        {
            auto n = named_ops[p.first];
            
            n->set_input_future_from_feed(p.second);
        }

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
        
        std::vector<tensor> output;
        output.reserve(fetches.size());
        
        for (auto const& p : fetches)
        {
            auto n = named_ops[p];
            
            n->get_output_future(0).then(
                hpx::util::unwrapped(
                    [&output](tensor t)
                    {
                        output.push_back(t);
                    }
                )
            );
        }
        
        hpx::wait_all(comp_futures);
                    
        return output;
    }
};
    
    
}} //namespace

#endif