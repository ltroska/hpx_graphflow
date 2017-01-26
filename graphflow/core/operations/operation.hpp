#ifndef HPX_GRAPHFLOW_OPERATION_HPP_
#define HPX_GRAPHFLOW_OPERATION_HPP_

#include "operation_context.hpp"
#include "core/tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace graphflow { namespace operations {
    
struct operation {
    inline std::vector<tensor> operator()(
            operation_context const& context, std::vector<tensor> input
        ) const
    {        
        return compute(context, input);
    }
    
    const std::size_t num_inputs() const { return n_inputs; }
    const std::size_t num_outputs() const { return n_outputs; }
    
protected:
    operation(std::size_t ni, std::size_t no) : n_inputs(ni) , n_outputs(no) {}
    
    virtual std::vector<tensor> compute(
            operation_context const& context, std::vector<tensor> input
        ) const = 0;
    
    const std::size_t n_inputs = 0;
    const std::size_t n_outputs = 0;
};

}} //namespace

#endif