#ifndef HPX_GRAPHFLOW_TENSOR_SHAPE_HPP_
#define HPX_GRAPHFLOW_TENSOR_SHAPE_HPP_

#include <cstdint>
#include <initializer_list>

namespace graphflow { 

class tensor_shape
{
    friend class tensor;
    
public:
    tensor_shape();
    tensor_shape(std::initializer_list<std::size_t> dims);
    
    inline int ndims() const { return ndims_byte(); }
    inline int num_elements() const { return num_elements_; }
    
    inline std::size_t operator[](std::size_t d) const { return dims_[d]; }
    
    void add_dim(int64_t size);
    
    inline bool is_scalar() const { return ndims() == 0; }
    inline bool is_vector() const { return ndims() == 1; }
    inline bool is_matrix() const { return ndims() == 2; }
    
    bool operator==(tensor_shape const& other) const;
    
private:

    inline uint8_t ndims_byte() const {return dims_[15];} 
    inline void set_ndims_byte(uint8_t b) {dims_[15] = b;} 

    uint8_t dims_[16];
    uint64_t num_elements_;
};

tensor_shape::tensor_shape() : num_elements_(1)
{
    set_ndims_byte(0);
}

tensor_shape::tensor_shape(std::initializer_list<std::size_t> dims)
: num_elements_(1)
{
    for (auto const& size : dims)
        add_dim(size);
}
    
void tensor_shape::add_dim(int64_t size)
{
    auto nd = ndims_byte();
    
    dims_[nd] = size;
    
    num_elements_ *= size;
    
    set_ndims_byte(nd + 1);
}

bool tensor_shape::operator==(tensor_shape const& other) const
{
    for (uint8_t i = 0; i < 14; ++i)
        if (dims_[i] != other.dims_[i])
            return false;
    
    return true;
}


} //namespace

#endif