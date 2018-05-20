This is an implementation for a linear, artificial neural network, in Common Lisp.
This project is supposed to be a port of a similar algorithm I once built using C++.

The implementation in Common Lisp was made in two days, and I used my own C++ code as
basis, though I stripped out most of the object-oriented programming aspect (except
encapsulation and methods, which works differently in CL -- and comes in handy!).
I also tried to keep imperative approaches and data mutation to a minimum, though
it relies quite a lot on it. I am not ashamed on that, since I did not want to put
many constraints on my coding style, but I do believe I did some clever functional
tricks wherever I could.

I also make extensible usage of arrays and the loop macro, so if you're allergic to
those, be warned.

Plus, I am not trying to necessarily formalize it as a Quicklisp package, but you can
clone/symlink it into you local-projects folder and load the 'ann system, and it will
work just fine. I hope.

Mind that, for now, the Eta and Alpha values of an ANN are global to the whole system,
and the transfer function is fixed to a simple f(x) = tanh x operation (which means
its derivative is f(x) = 1 - x^2).

This project is distributed under the MIT License.

-----

Here is a small documentation for the meaningful parts of this project:


* package "ann"
This is the core package of the system. Contains the whole implementation of the
artificial neural network.


** *neuron-eta*
[variable] Overall network training rate. Must be a number [0.0..1.0]. 
0.0 is a slow learner; 0.2 is a medium learner; 1.0 is a reckless learner.
Defaults to 0.15.

** *neuron-alpha*
[variable] Weight-changing momentum for the inter-layer connections between neurons.
Must be a number [0.0..1.0]. 0.0 is no momentum, 0.5 is moderate momentum.
Defaults to 0.5

** ann
[struct] Represents a whole artificial neural network, with its internal data.

** neuron
[struct] Represents a single neuron in a layer of the artificial neural network.

** connection
[struct] Represents a single connection between a neuron from layer A to another
neuron to the next layer B.

** (build-ann topology)
[function] Builds an artificial neural network using a topology, given as a list
of natural numbers. For example, a topology '(2 4 1) yields a neural network with
two input neurons and one output neuron, and a single hidden layer containing four
neurons. Each layer will also have a single extra bias neuron at end.

** (feed-forward ann input-values)
[method] Feeds forward the given input values to the neural network, effectively
performing the pattern recognition. The values fed must be a list of single-float
values which pair up with each input neuron -- e.g. a neural network with two
input neurons must also be fed a list of two single-floats, like '(1.0 0.0).
The bias neuron of the input layer is disregarded as input; and if needed, remember
to coerce each input value to a float.

** (back-propagate ann target-values)
[method] Backpropagates an expected result after a successful feed-forward operation,
effectively training the artificial neural network to better recognize the pattern
for the last executed test case. The target values given must be a list of single-float
values which pair up with each output neuron -- e.g. a neural network with one output
neuron contains yields a single result number, and so the target value must be a list
containing one single-float, like '(1.0).
The bias neuron of the output layer is disregarded as target; and if needed, remember
to coerce each target value to a float.

** (collect-results ann)
[method] Yields a list, containing the results that the neural network yields after
processing the last fed-forward data. This results in a list of single-floats, containing
the output value for every neuron on the output layer, except the bias neuron's.

** (run-training ann input-test-cases target-test-cases &optional show-output)
[method] Performs automated training cycle comprised of feeding forward each test case
and backpropagating the target for that test case. The input cases must be a list of lists;
each sublist must be a possible and valid input for a feed-forward operation. The target test
cases are also a list of lists, and must also be comprised of possible and valid inputs for a
backpropagation operation -- e.g. for a single input case (1.0, 0.0) which expects a target
1.0, one can create a list '((1.0 0.0)) of input tests and a list '((1.0)) of target tests.
One can also specify whether outputting the current test case and the yielded value is also
needed by specifying show-output.
The whole training process is monitored using the time function, so at the end of the training,
the function will yield information regarding execution time, CPU usage and etc.




* package "ann-test"
This package contains tests for the overall artificial neural network implementation.


** *xor-ann*
[parameter] When not nil, contains an instantiation of a neural network which is supposed
to work as a test for learning the exclusive-or operation. Since we cannot simulate bits
on this ANN implementation, we use 1.0 and 0.0 as 1 and 0.

** (xor-begin)
[function] Creates a neural network on parameter *xor-ann* and generates test cases for
training it in exclusive-or operations.
By default, this function generates a neural network with topology (2 4 1) and 20,000
test cases.

** (xor-train &optional show-input)
[function] Performs training on *xor-ann* so it can perform exclusive-or operations,
then releases all generated test cases for garbage collection.
Since this function uses the default training method defined on the artificial neural
network infrastructure, one can also opt to show the results for each backpropagated
test case on console by specifying it through show-input.
This function will perform nothing unless *xor-ann* is not nil and the test cases are
properly generated.

** (xor-finish)
[function] Performs a last test, this time with no training whatsoever, only by feeding
forward the four possible cases of the exclusive-or operation and showing them onscreen
in a readable way, along with the yielded output values.
This function will perform nothing unless *xor-ann* is not nil.

** (xor-run-test &optional show-input)
[function] Performs the whole creation and training cycle for the artificial neural
network which simulates the exclusive-or operation.
One can also opt to show the results of each training test case on console by specifying
it through show-input.
