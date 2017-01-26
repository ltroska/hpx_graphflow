#ifndef HPX_GRAPHFLOW_CORE_OPERATIONS_CONSTANT_OP_HPP_
#define HPX_GRAPHFLOW_CORE_OPERATIONS_CONSTANT_OP_HPP_

#include "core/tensor/tensor.hpp"
#include "operation_context.hpp"
#include "operation.hpp"

namespace graphflow { namespace operations {
    
struct constant_op : public operation
{   
    constant_op(tensor tens) : operation(1, 1), t(tens) {}
    
    std::vector<tensor> compute(
            operation_context const& context, std::vector<tensor> input
        ) const override
    {               
        if (input.size() != 0)
            return input;
        return {t};
    }
    
private:
    tensor t;
};
    
}} //namespace

#endif