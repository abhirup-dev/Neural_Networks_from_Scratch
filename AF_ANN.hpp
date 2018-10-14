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
#include "Layer.hpp"
using std::vector;
using std::string;
class Net{
public:
	int num_layers;
	vector<Layer> network;
	vector<af::array> weights;
	vector<int> topology;
	vector<string> activations;
	Net(const vector<int> &topo, const vector<string> activs, double range);
	void setNetwork(const vector<vector<float>> &inputVals);
	void feedForward();
	void backProp(const vector<vector<float>> &targetVals, float lr_rate);
	void getLoss(const af::array &targetVals);
	vector<float> getOutput();
	void print_all_layers();
	void weight_dims();
	af::array net_loss;
	float alpha;
};
void Net::print_all_layers()
{
	for(auto &i: network)
		af_print(i.layer);
}
void Net::weight_dims()
{class Layer;
class Net;
	std::cout << "Weights:\n";
	for(auto &w: weights)
		std::cout << w.dims() << "\n";
}
Net::Net(const vector<int> &topo, const vector<string> activs, double range)
{
	topology = topo;
	activations = activs;
	num_layers = topology.size();
	network.resize(topology.size());
	weights.resize(topology.size() - 1);
	for(int i=0; i<topo.size()-1; i++)
		weights[i] = range * af::randu(topo[i]+1, topo[i+1]) - range/2;
	net_loss = 0;
}
void Net::setNetwork(const vector<vector<float>> &inputVals)
{
	assert ((inputVals[0].size() == topology[0]) && "Invalid inputVals dimensions.");
	int b_size = inputVals.size();
	for(int i=0; i<num_layers; i++)
		if(i != num_layers-1){
			network[i] = Layer(b_size, topology[i]+1, activations[i]);
			network[i].layer(af::span, topology[i]) = 1.0;
		}
		else
			network[i] = Layer(b_size, topology[i], activations[i]);
	
	for(int i=0; i<inputVals.size(); i++)
		for(int j=0; j<inputVals[0].size(); j++)
			network[0].layer(i,j) = inputVals[i][j];

}
void Net::feedForward()
{
	for(int i=0; i<num_layers-1; i++)
	{
		Layer &ll = network[i];
		af::array tmp = af::matmul(ll.layer, weights[i]);
		network[i+1].layer(af::span, af::seq(tmp.dims(1))) = ll.activ_fn(tmp);
	}
}
void Net::backProp(const vector<vector<float>> &targetVals, float lr_rate)
{
	af::array out = network.back().layer;
	assert (targetVals.size() == out.dims(0) && targetVals[0].size() == out.dims(1) && "Invalid targetVals dimensions.");
	alpha = lr_rate;//learning rate

	int batch_size = targetVals.size();
	af::array target(targetVals.size(), targetVals[0].size());
	for(int i=0; i<targetVals.size(); i++)
		for(int j=0; j<targetVals[0].size(); j++)
			target(i,j) = targetVals[i][j];

	//func.pointer 'deriv' points to appropriate activation fn.
	//... for respective layers
	af::array (*deriv)(const af::array&) = network.back().activ_deriv_fn;

	net_loss = out - target;
	af::array err = net_loss;

	for(int i=num_layers-2; i>=0; i--)
	{
		Layer &curr = network[i];

		af::array delta = (deriv(out)*err).T();

		//adjusting Weights
		af::array grad = -(alpha * af::matmul(delta, curr.layer)) / batch_size;
		weights[i] += grad.T();

		//'out' will denote current layer in next step of backprop
		//'out' excludes bias of current layer
		out = curr.layer(af::span, af::seq(curr.layer.dims(1)-1));
//		//updating the activation function for 'out'
		deriv = curr.activ_deriv_fn;
//
		err = af::matmulTT(delta, weights[i]);
//		//Removing error bias
		err = err(af::span, af::seq(out.dims(1)));
	}
}
vector<float> Net::getOutput()
{
	vector<float> out(network.back().layer.elements());
	network.back().layer.host(out.data());
	return out;
}

