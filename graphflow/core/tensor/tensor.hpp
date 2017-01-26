#ifndef HPX_GRAPHFLOW_TENSOR_HPP_
#define HPX_GRAPHFLOW_TENSOR_HPP_

#include "tensor_shape.hpp"
#include "core/util/random.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

#include <hpx/runtime/serialization/serialize_buffer.hpp>

namespace graphflow {
    
class tensor
{
    typedef hpx::serialization::serialize_buffer<double> buffer_type;
    
    template<typename T>
    struct array_deleter {
        void operator()(T const* p)
        {
            delete [] p;
        }
    };
    
public:
    tensor() : data_(1) {};
    tensor(std::initializer_list<std::size_t> dims)
    : shape(dims),
        data_(new double[shape.num_elements()], shape.num_elements(),
                buffer_type::take, array_deleter<double>()
            )
    {}
        
    tensor(Eigen::MatrixXd const& mat)
    {
        shape.add_dim(mat.rows());
        shape.add_dim(mat.cols());
        
        data_ = buffer_type(new double[shape.num_elements()],
                            shape.num_elements(), buffer_type::take,
                            array_deleter<double>());
        
        Eigen::Map<Eigen::MatrixXd>(get_data(), mat.rows(), mat.cols()) = mat;
    }
        
    void fill_random()
    {
        for (auto& v : data_)
            v = util::random(0, 100);
    }
                
    inline double* get_data() { return data_.data(); }
    
    inline double& operator[](std::size_t index) { return data_[0]; }
            
    tensor_shape shape;
        
private:
    buffer_type data_;
};

   
}

#endif