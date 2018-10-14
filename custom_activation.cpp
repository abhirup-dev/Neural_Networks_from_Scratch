// *************************************************************
// Author: Abhirup Das
// Github: https://github.com/codebuddha
// LinkedIn: https://www.linkedin.com/in/abhirup-das-5a174212a/
// *************************************************************
#include <arrayfire.h>
#include <af/util.h>
#include <iostream>
#include "Layer.hpp"

af::array ReLU (const af::array &x)
{
    af::array y = x;
    y(af::where(y<0)) = 0;
    return y;
}
af::array deriv_ReLU(const af::array &x)
{
    af::array y = x;
    y(af::where(y<0)) = 0;
    y(af::where(y==0)) = 0.5;
    y(af::where(y>0)) = 1.0;
    return y;
}
int main()
{
    auto &activs = Layer::activ_map;
    float a[] = {-1, -0.5, 0, 0.5, 0, 1};
    af::array input(5, a);
    std::cout << "List of already included Activation functions:\n";
    for(auto &i: activs)
    {
        std::cout << "Name: " << i.first <<"\n";
        af_print(i.second.first(input));
    }
    std::cout << "Including custom ReLU activation in Layer::activ_map.\n";
    Layer::setNewActivation(&ReLU, &deriv_ReLU, std::string("custom_ReLU"));
    std::cout << "Updated Activations list.\n";
    for(auto &i: activs)
        std::cout << i.first << "\n";
    std::cout << "Now it can be used like other included activation functions in 'Layer.hpp'\n";
}