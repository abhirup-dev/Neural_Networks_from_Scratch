// *************************************************************
// Author: Abhirup Das
// Github: https://github.com/codebuddha
// LinkedIn: https://www.linkedin.com/in/abhirup-das-5a174212a/
// *************************************************************
#include <iostream>
#include <arrayfire.h>
#include <af/util.h>
#include <af/gfor.h>
#include <cmath>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <string>
#include "AF_ANN.hpp"
float calc_LRdecay(float lr, int step, float div, int epoch)
{
	lr *= powf(div, int(epoch/step));
	return lr;
}
int main()
{
	// std::cout << "started\n";
	vector<int> topo = {3, 2, 1};
	vector<string> activations = {"tanh", "tanh", "tanh", "tanh"};
	Net nn = Net(topo, activations, 2);

	vector<vector<float>> inputs = {
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 0},
			{0, 1, 1},
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 0},
			{1, 1, 1}
	}, targets = {
		{0},
		{1},
		{1},
		{0},
		{1}, 
		{0}, 
		{0}, 
		{1}
	};
	assert(inputs.size() == targets.size() && "Check data dimensions.");
	
	// std::cout <<calc_LRdecay(0.1, 100, 0.5, 101) << "\n";
	// std::cout <<calc_LRdecay(0.1, 100, 0.5, 201) << "\n";
	nn.setNetwork(inputs);
	float lr_rate = 0.12;
	int batch_size = inputs.size();
	for(int i=0; i<=50000; i++)
	{
		nn.feedForward();
	//  decaying lr_rate  every 10000 steps by 0.95 times
		nn.backProp(targets, calc_LRdecay(lr_rate, 10000, 0.95, i));
		if(i % 1000 == 0)
			af_print(nn.net_loss);
		if(i % 1000 == 0)
			std::cout << "==========" << i << "==========\n";
	}

	vector<vector<float>> in ={{0, 1, 0}, {0, 0, 0}}, out ={{1}, {0}};
	nn.setNetwork(in);
	nn.feedForward();
	auto res = nn.getOutput();
	std::cout << "Results: ";
	for(auto &o: res)
		std::cout << o << " ";
	std::cout << "\n";
}