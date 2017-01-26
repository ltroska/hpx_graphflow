#include "graphflow.hpp"

#include <hpx/hpx_init.hpp>
#include <Eigen/Core>

#include <iostream>

using namespace graphflow;

std::shared_ptr<graph::node> MatMul(graph::graph& g,
    std::shared_ptr<graph::node> A, std::shared_ptr<graph::node> B
    )
{
    auto r = g.add_node();    
    r->set_operation(std::make_shared<operations::mat_mul_op>());
    g.add_edge(A, 0, r, 0);
    g.add_edge(B, 0, r, 1);
    
    return r;
}

std::shared_ptr<graph::node> Const(graph::graph& g, tensor t)
{
    auto r = g.add_node();    
    r->set_operation(
        std::make_shared<operations::constant_op>(operations::constant_op(t))
    );

    return r;
}

int hpx_main(int argc, char* argv[])
{
    tensor random_tensor{5, 5};
    random_tensor.fill_random();
            
    tensor identity(Eigen::MatrixXd::Identity(5, 5));
    tensor two_identity(Eigen::MatrixXd::Identity(5, 5) * 2);
    tensor five_identity(Eigen::MatrixXd::Identity(5, 5) * 5);
            
    auto g = graph::graph();
            
    auto A = Const(g.with_op_name("A"), identity);
    auto F = Const(g.with_op_name("F"), five_identity);
    auto B = Const(g.with_op_name("B"), random_tensor);
    auto C = MatMul(g.with_op_name("C"), A, B);
    auto D = MatMul(g.with_op_name("D"), C, A);
    auto E = MatMul(g.with_op_name("E"), C, F);
    
    session s;
        
    std::vector<tensor> outputs;
    
    std::vector<std::pair<std::string, tensor> > feeds
        {{"A", two_identity}};
        
    std::vector<std::string> fetches
        {"A", "B", "C", "D", "E"};
        
    s.run(g, feeds, fetches, outputs);
    
    for (auto& t : outputs)
        std::cout
            << Eigen::Map<Eigen::MatrixXd>(t.get_data(), t.shape[0], t.shape[1])
            << std::endl << std::endl;
        
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}