#ifndef HPX_GRAPHFLOW_CORE_OPERATIONS_MAT_MUL_OP_HPP_
#define HPX_GRAPHFLOW_CORE_OPERATIONS_MAT_MUL_OP_HPP_

#include "core/tensor/tensor.hpp"
#include "operation.hpp"

#include <Eigen/Core>

namespace graphflow { namespace operations {
    
struct mat_mul_op : public operation
{   
    mat_mul_op() : operation(2, 1) {}
    
    std::vector<tensor> compute(
            operation_context const& context, std::vector<tensor> input
        ) const override
    {        
        HPX_ASSERT(input.size() == 2);
        
        auto& A = input[0]; 
        auto& B = input[1];
        
        HPX_ASSERT(A.shape.is_matrix() && B.shape.is_matrix());
      
        tensor C(
            Eigen::Map<Eigen::MatrixXd>(A.get_data(), A.shape[0], A.shape[1])
            *
            Eigen::Map<Eigen::MatrixXd>(B.get_data(), B.shape[0], B.shape[1])
        );        
        
        std::vector<tensor> output{std::move(C)};
                
        return output;
    }
};
    
}} //namespace

#endif