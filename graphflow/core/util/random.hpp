#ifndef HPX_GRAPHFLOW_CORE_UTIL_RANDOM_HPP_
#define HPX_GRAPHFLOW_CORE_UTIL_RANDOM_HPP_

#include <random>
#include <functional>
#include <algorithm>

namespace graphflow { namespace util {
   
template<typename Numeric, typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to)
{
    thread_local static Generator gen(std::random_device{}());

    using dist_type = typename std::conditional
    <
        std::is_integral<Numeric>::value
        , std::uniform_int_distribution<Numeric>
        , std::uniform_real_distribution<Numeric>
    >::type;

    thread_local static dist_type dist;

    return dist(gen, typename dist_type::param_type{from, to});
} 
    
}} //namespace

#endif