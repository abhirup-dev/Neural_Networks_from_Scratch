// *************************************************************
// Author: Abhirup Das
// Github: https://github.com/codebuddha
// LinkedIn: https://www.linkedin.com/in/abhirup-das-5a174212a/
// *************************************************************
#include <iostream>
#include <vector> 
#include <cstdlib>
#include <cassert>
#include <cmath>
class Neuron;
using Layer = std::vector<Neuron>;
class Net;
// ==================================================================
// class Connection 
struct Connection
{
	double weight;
	double delWeight;
};
// ==================================================================
// class Neuron
class Neuron
{
public:
	unsigned numOutputs;
	Neuron(unsigned num, int index);
	void setVal(double x){Val = x;}
	double getVal(void) const{return Val;}
	int getIndex(void) const{return layer_index;}
	void feedforward_from(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	double Val;
	double layer_index;
	double gradient;
	std::vector<Connection> outputWeights;
	double sumDOW(const Layer &nextLayer)const;
	static double tanh_activation(double x){return tanh(x);}
	static double tanh_deriv_activation(double x){return 1.0 - x*x;}
	static double randomWeight(void){return rand() / double(RAND_MAX);}
	static double eta; //learning rate
	static double alpha; //momentum
	
}; 
double Neuron::eta = 0.5;
double Neuron::alpha = 0.2;
Neuron::Neuron(unsigned num, int index)
{
	numOutputs = num;
	outputWeights.resize(numOutputs);
	for(auto &i : outputWeights)
		i.weight = randomWeight();
	layer_index = index;
}
void Neuron::feedforward_from(const Layer &prevLayer)
{
	double sum=0;
	for(int i=0; i<prevLayer.size(); i++)
		sum += prevLayer[i].Val * prevLayer[i].outputWeights[layer_index].weight;
	Val = tanh_activation(sum);
}
void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - Val;
	gradient = delta * Neuron::tanh_deriv_activation(Val);
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::tanh_deriv_activation(Val);
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	for(int i=0; i<nextLayer.size(); i++)
		sum += nextLayer[i].gradient * outputWeights[i].weight;
	return sum;
}
void Neuron::updateInputWeights(Layer &prevLayer)
{
	for(auto &neuron: prevLayer)
	{
		double oldDelWeight = neuron.outputWeights[layer_index].delWeight;
		double newDelWeight = eta * neuron.Val * gradient 
								+ alpha * oldDelWeight;
		neuron.outputWeights[layer_index].delWeight = newDelWeight;
		neuron.outputWeights[layer_index].weight += newDelWeight;
	}
}
// ==================================================================
// class Net for defining the network
class Net
{
public:
    Net(const std::vector<unsigned> &topology);
    void feedforward(const std::vector<double> &inputVals);
    void backprop(const std::vector<double> &targetVals);
	void calcLoss(const std::vector<double> &targetVals);
    void getResults(std::vector<double>resultVals) const;
	void predict(const std::vector<double> &testInput);
	void printConfig()
	{
		std::cout << "Layer config:" << "\n";
		for(auto &layer: net_layers)
			std::cout << layer.size() << " ";
		std::cout << "\n";
	}
	void getWeights()
	{
		std::cout << "Weights:" << "\n";
		for(auto &layer : net_layers)
			for(auto &neuron: layer)
			{	
				std::cout << neuron.numOutputs << " index:" << neuron.getIndex() << "\tval:" << neuron.getVal();
				std::cout << "\n";
			}
	}
	double net_error;
private:
	std::vector<Layer> net_layers; //net_layers[layer_num][neuron_num]
};

Net::Net (const std::vector<unsigned> &topology)
{ 
	unsigned numLayers = topology.size();
	net_layers.resize(numLayers); 
	fill(net_layers.begin(), net_layers.end(), Layer());
	for(int i=0; i<topology.size(); i++)
	{
		unsigned numOutputs = i==topology.size()-1? 0: topology[i+1]; 
		for(int j=0; j<=topology[i]; j++)
			net_layers[i].push_back(Neuron(numOutputs, j));
		net_layers[i].back().setVal(1.0);
	}
}
void Net::feedforward(const std::vector<double> &inputVals)
{
	assert(inputVals.size() == net_layers[0].size()-1);
	for(int i=0; i<inputVals.size(); i++)
		net_layers[0][i].setVal(inputVals[i]);
	for(int i=1; i<net_layers.size(); i++)
	{
		Layer &prevLayer = net_layers[i-1], &currLayer = net_layers[i];
		for(int j=0; j<currLayer.size()-1; j++)
			currLayer[j].feedforward_from(prevLayer);
	}
}
void Net::backprop(const std::vector<double> &targetVals)
{
	Layer &last_layer = net_layers.back();
	// Compute gradients for the output Layer
	for(int i=0; i<last_layer.size()-1; i++)
		last_layer[i].calcOutputGradients(targetVals[i]);

	// Compute gradients for the Hidden layers
	for(int i=net_layers.size()-2; i>0; i--)
	{
		Layer &nextLayer = net_layers[i+1];
		for(auto &neuron: net_layers[i])
			neuron.calcHiddenGradients(nextLayer);
	}

	// Update all connection weights as required
	for(int i=net_layers.size()-1; i>0; i--)
	{
		Layer &prevLayer = net_layers[i-1];
		for(auto &neuron: net_layers[i])
			neuron.updateInputWeights(prevLayer);
	}
}
void Net::calcLoss(const std::vector<double> &targetVals)
{
	Layer &last_layer = net_layers.back();
	assert (targetVals.size() == last_layer.size()-1);

	// Compute the RMS error for the output Layer
	net_error = 0.0;
	for(int i=0; i<targetVals.size(); i++)
		net_error += pow(last_layer[i].getVal() - targetVals[i], 2);
	net_error /= targetVals.size();
	net_error = sqrt(net_error);
	// std::cout << "RMS error = " << net_error << "\n";
}
void Net::predict(const std::vector<double> &testInput)
{
	feedforward(testInput);
	Layer &results = net_layers.back();
	std::cout << "Prediction: ";
	for(int i=0; i<results.size()-1; i++)
		std::cout << results[i].getVal() << " ";
	std::cout << "\n";
}
int main()
{
    std::vector<unsigned> topology{3,2,2,1}; 
	int num_inputs = topology[0], num_outputs = topology.back();
	int num_samples = 4;
    Net myNN(topology);
	myNN.printConfig();
	// myNN.getWeights();
	std::vector<std::vector<double>> inputVals(4, std::vector<double>(num_inputs));
	std::vector<std::vector<double>> targetVals(4, std::vector<double>(num_outputs));;
	std::cout <<"Enter training data\n";
	for(int i=0; i<num_samples; i++)
	{
		for(auto &j: inputVals[i])
			std::cin >> j;
		for(auto &j: targetVals[i])
			std::cin >> j;
	}
	int n_epoch = 5000000;
	for(int i=1; i<n_epoch; i++)
	{
		for(int j=0; j<num_samples; j++)
		{	
			myNN.feedforward(inputVals[j]);
			myNN.calcLoss(targetVals[j]);
			if(i%100000 == 0)
				std::cout << "epoch:" << i << "\t" << myNN.net_error << "\n";
			myNN.backprop(targetVals[j]);
		}
	}

	std::cout << "Test Now..\n"; 
	std::vector<double> testInput(num_inputs), testTarget(num_outputs);
	do{
		std::cout <<"Test model\n";
		for(auto &i: testInput)
			std::cin >> i;
		myNN.predict(testInput);
	}while(true);

}
