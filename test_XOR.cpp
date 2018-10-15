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
af::array vec_to_AF(std::vector<std::vector<float>> &arr)
{
	af::array arr2(arr.size(), arr[0].size());
	for(int i=0; i<arr.size(); i++)
		for(int j=0; j<arr[0].size(); j++)
			arr2(i,j) = arr[i][j];
	return arr2;
}
int main()
{
	//Inputs:3(tanh)
	//Hidden:2(tanh)
	//Output:1(tanh)
	vector<int> topo = {3, 5, 1};
	vector<string> activations = {"none", "sigmoid", "sigmoid"};
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

	af::array af_inputs = vec_to_AF(inputs), af_targets = vec_to_AF(targets);
	nn.setNetwork(af_inputs);
	float lr_rate = 0.1, multi=1; int steps=10000;
	int batch_size = inputs.size();
	// af::Window w1(512, 512);
	// af::Window w2(1000, 1000);
	// af::Window w3(700, 700);
	// af::array plotter(501, 2);
	for(int i=0; i<=100000; i++)
	{
		af::array loss = af::sqrt(af::sum(af::pow(nn.net_loss, 2))/batch_size);

		// for(int j=2; j<3; j++)
		// {
		// 	char c[] = {'A', 'b', char(j+1)};
		// 	af::Window &win = w3;
		// 	af::array wh = af::histogram(af::flat(nn.weights[j]), 60); 
		// 	win.hist(wh, 0, 60, c);
		// }
		nn.feedForward();	
		nn.L1loss(af_targets);
		if(i % 5000 == 0)
		{
			af_print(nn.getOutput());
			std::cout << "===" << i << "===" << "lr@" << calc_LRdecay(lr_rate, steps, multi, i) << "===\n";
		}
		// if(i % 1000 == 0)
		//  decaying lr_rate every 10000 steps by 0.95 times
		nn.backProp(calc_LRdecay(lr_rate, steps, multi, i));
	}
	// visualisation with ArrayFire
	// af::Window window(512, 512);
	// do{
	// 	window.plot(plotter(af::span,0), plotter(af::span,1));
	// }while(!window.close());
	// vector<vector<float>> in ={{0, 1, 0}, {0, 0, 0}}, out ={{1}, {0}};
	// nn.setNetwork(vec_to_AF(in));
	// nn.feedForward();
	// auto res = nn.getOutput();
	// std::cout << "Results: ";
	// for(auto &o: res)
	// 	std::cout << o << " ";
	// std::cout << "\n";
}