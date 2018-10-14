# Neural Networks from Scratch
Implementing Artificial Neural Network from scratch using C++

There are two implementations listed here:
1) A [basic_ann](https://github.com/codebuddha/Neural_Networks_from_Scratch/blob/master/basic_ann.cpp) without using any libraries, just raw C++. The [main](https://github.com/codebuddha/Neural_Networks_from_Scratch/blob/55c7b0e9a8a3571a726ab744151ba351a4840dfb/basic_ann.cpp#L191) function here is to train the network to act as a 3-input XOR operator.

2) A vectorized implementation of an ANN using the [ArrayFire](http://arrayfire.org/docs/index.htm) library that allows the use of an unified source-code to compile programs for CPU, CUDA, as well as OpenCL.

Some features that allow for smooth experimentation:
    - Ability to add custom activation functions using the [`Layer::setNewActivation`]() function, as shown [here]().