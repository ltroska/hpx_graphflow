#ifndef HPX_GRAPH_FLOW_CORE_GRAPH_GRAPH_HPP_
#define HPX_GRAPH_FLOW_CORE_GRAPH_GRAPH_HPP_

#include "core/operations/operation.hpp"

#include "hpx/lcos/local/promise.hpp"

namespace graphflow { namespace graph {

class node;
class graph;
    
class edge
{
    friend node;
    friend graph;
    
public:
    edge(std::shared_ptr<node> src_node, std::size_t src_node_slot,
            std::shared_ptr<node> dst_node, std::size_t dst_node_slot)
    : src(src_node), src_slot(src_node_slot), dst(dst_node),
        dst_slot(dst_node_slot)
    {}
    
protected:
    std::shared_ptr<node> src; 
    std::shared_ptr<node> dst;
    std::size_t src_slot;
    std::size_t dst_slot;
};
    
class node
{
    friend graph;
    
public:
    node(std::size_t i, std::string name = "") : id(i), op_name(name) {}
    
    void set_operation(std::shared_ptr<operations::operation> o)
    {
        op = o;
        
        setup_dependencies(op->num_inputs(), op->num_outputs());
    }
    
    void setup_dependencies(std::size_t num_inputs, std::size_t num_outputs)
    {
        input_futures.reserve(num_inputs);
        
        output_promises.resize(num_outputs);
        output_futures.reserve(output_promises.size());
        
        for (auto& p : output_promises)
            output_futures.push_back(p.get_future().share());
    }
    
    inline std::shared_ptr<operations::operation> get_operation() {return op;}
    
    inline std::size_t get_id() const { return id;}
    
    inline std::vector<std::shared_ptr<edge> > const& get_outedges()
    {
        return outedges;
    }
    
    inline std::vector<std::shared_ptr<edge> > const& get_inedges()
    {
        return inedges;
    }
    
    void set_output_promises(std::vector<tensor> output)
    {
        for (std::size_t i = 0; i < output.size(); ++i)
            output_promises[i].set_value(output[i]);
    }
    
    inline hpx::shared_future<tensor> get_output_future(std::size_t index)
    {        
        return output_futures[index]; 
    }
    
    inline void set_all_input_futures()
    {
        for(auto& e : inedges)
            input_futures.push_back(e->src->get_output_future(e->src_slot));
    }
    
    inline void set_input_future_from_feed(tensor t)
    {
        input_futures.push_back(hpx::make_ready_future(t));
    }
    
    inline std::vector<hpx::lcos::shared_future<tensor> > const&
        get_input_futures()
    {
        return input_futures;
    }
    
    inline std::string const& get_op_name() const { return op_name; }
            
protected:    
    std::size_t id;
    std::string op_name;
        
    std::shared_ptr<operations::operation> op;
    
    std::vector<std::shared_ptr<edge> > outedges;
    std::vector<std::shared_ptr<edge> > inedges;

    void add_outedge(std::shared_ptr<edge> e) {outedges.push_back(e);}
    void add_inedge(std::shared_ptr<edge> e) {inedges.push_back(e);}
    
    std::vector<hpx::lcos::local::promise<tensor> > output_promises;
    std::vector<hpx::lcos::shared_future<tensor> > output_futures;
    std::vector<hpx::lcos::shared_future<tensor> > input_futures;    
};
    
class graph
{
public: 
    graph() : root(nullptr), running_id(0)
    {
        root = std::make_shared<node>(running_id++);
    }

    inline std::shared_ptr<node> get_root() const
    {
        return root;
    }
    
    std::shared_ptr<node> add_node()
    {
        auto n = std::make_shared<node>(running_id++, op_name);
        nodes.push_back(n);
        
        if (op_name != "")
        {
            named_op_map[op_name] = n;
            op_name = "";
        }
                        
        return n;
    }
    
    void add_edge(std::shared_ptr<node> src, std::size_t src_slot,
                    std::shared_ptr<node> dst, std::size_t dst_slot)
    {
        auto e = std::make_shared<edge>(src, src_slot, dst, dst_slot);
        edges.push_back(e);
        src->add_outedge(e);
        dst->add_inedge(e);
    }
    
    std::vector<std::shared_ptr<node> > const& get_nodes() const
    {
        return nodes;
    }
    
    std::vector<std::shared_ptr<edge> > const& get_edges() const
    {
        return edges;
    }
    
    void setup_dependencies()
    {
       // root->setup_dependencies(0, num_feeds);
        
        for (auto& n : nodes)
            n->set_all_input_futures();
    }
    
    std::vector<std::shared_ptr<node> > get_sinks() const
    {
        std::vector<std::shared_ptr<node> > result;
        
        for (auto& n : nodes)
            if (n->get_outedges().size() == 0)
                result.push_back(n);
                
        return result;
    }
    
    graph& with_op_name(std::string name)
    {
        op_name = name;
        return *this;
    }
    
    std::unordered_map<std::string, std::shared_ptr<node> >
        get_named_ops() const
    {
        return named_op_map;
    }
            
protected:
    std::size_t running_id;
    std::shared_ptr<node> root;
    
    std::vector<std::shared_ptr<node> > nodes;    
    std::vector<std::shared_ptr<edge> > edges;
    
    std::unordered_map<std::string, std::shared_ptr<node> > named_op_map;

    std::string op_name = "";
};
    
}} // namespace

#endif