# This file contains the Value class, which allow backpropagation and creation
# of auto-labelled graphs, as well as the implementation of the MLP.
# An example of usage is shown in main()





import math
from graphviz import Digraph
import random





class Value:
    # Attributes
    should_show_labels = False
    def __init__(self, data, label="", _children=(), _op=""):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0
        self.should_create_new_labels = False

    # Toggles Value.should_show_labels, which is a boolean
    @classmethod
    def toggle_label_display(cls):
        cls.should_show_labels = (not cls.should_show_labels)

    # Decides what self.should_create_new_labels should be, which is a boolean
    def set_new_label_creation(self, other):
        if isinstance(other, Value):
            self.should_create_new_labels = True if (len(self.label) + len(other.label)) < 21 and self.label != "" and other.label != "" else False
        else:
            self.should_create_new_labels = True if (len(self.label) + other) < 21 and self.label != "" else False 

    # Returns an f-string for print(self)
    def __repr__(self):
        return f"Value(\"{self.label}\", data={self.data:.3g})" if (Value.should_show_labels and self.label != "") else f"Value(data={self.data:.3g})"

    # Creates an svg-Digraph of the expression of Value objects leading up to self 
    def create_graph(self):
        
        # Build a set of known Value objects and edges between them,
        # starting from a Root Value object 
        def trace(root_node):
            nodes, edges = set(), set()
            def build(parent_node):
                if parent_node not in nodes:
                    nodes.add(parent_node)
                    for child_node in parent_node._prev:
                        edges.add((child_node, parent_node))
                        build(child_node)
            build(root_node)
            return nodes, edges

        # Returns the completed Digraph
        def graph(root_node):

            # Get the set of known Value objects and the edges between them,
            # starting from a Root Value object 
            nodes, edges = trace(root_node)
            
            # Create the empty Digraph
            dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'bgcolor':'#000c18'}, filename="my_graph")

            # For each known Value object, create a node for it
            for node in nodes:
                node_name = str(id(node))
                full_label = f"{node.label} | data {node.data:.3g} | grad {node.grad:.3g}"
                # displays_full_label = node.label != "" and (node.label != f"{node.data:.3g}")
                # final_label = full_label if displays_full_label else f"data {node.data:.3g} | grad {node.grad:.3g}"
                dot.node(name=node_name, label=full_label, shape="record", color="#6688cc", fontcolor="#6688cc")
                
                # If that Value object was created by an operation,
                # create an operation node that reflects that and
                # connect it to the Value object's node
                if node._op:
                    dot.node(name=(node_name + node._op), label=node._op, color="#6688cc", fontcolor="#6688cc")
                    dot.edge((node_name + node._op), node_name, color="#6688cc")

            # For each known edge between a Child and a Parent Value object,
            # connect the Child's node to the Parent's operation node
            for child_node, parent_node in edges:
                dot.edge(str(id(child_node)), (str(id(parent_node)) + parent_node._op), color="#6688cc")

            # Return the completed Digraph
            return dot

        # Create the Digraph, making self be the Root Value object 
        g = graph(self)
        # Render the svg-Digraph to a folder
        g.render(directory="graphviz_outputs")

    def backward(self):
        # Treat self as the Root node (or the Root Value object), and backpropagate through it

        # Build a topological sorting of the nodes used to create self
        topo_sorted = []
        seen = set()
        def explore(parent_node):
            if parent_node not in seen:
                seen.add(parent_node)
                for child_node in parent_node._prev:
                    explore(child_node)
                topo_sorted.append(parent_node)
        explore(self)

        # dself/dself = 1
        self.grad = 1
        # Call the _backward of each node in the reverse order of our topological sorting
        # Remember, all _backward does is set the Value.grad attributes of children nodes
        for node in reversed(topo_sorted):
            node._backward()

    # Value + Value or Value + int/float
    def __add__(self, other):
        # Make sure other is a Value instance
        other = other if isinstance(other, Value) else Value(other, label=f"{other:.3g}")

        # Create the out Value instance
        out = Value(data=(self.data + other.data), _children=(self, other), _op="+")

        # Create out's label
        l = f"{self.label} + {other.label}"
        self.set_new_label_creation(other)
        out.label = l if self.should_create_new_labels else f"{out.data:.3g}"

        # Create out's _backward behavior
        def _backward():
            # Local derivatives
            dout_dself = 1.0
            dout_dother = 1.0
            # Global derivatives
            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother
        # Give out its _backward behavior
        out._backward = _backward

        return out
    
    # Called when int/float + Value
    # int/float is other
    # Value is self
    def __radd__(self, other):
        return self + other
    
    # Value*Value or Value*int/float
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other:.3g}")

        out = Value(data=(self.data * other.data), _children=(self, other), _op="*")

        l = f"({self.label})*{other.label}"
        self.set_new_label_creation(other)
        out.label = l if self.should_create_new_labels else f"{out.data:.3g}"

        def _backward():
            dout_dself = other.data
            dout_dother = self.data
            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother
        out._backward = _backward
        
        return out

    # int/float * Value
    def __rmul__(self, other):
        return self * other

    # -Value 
    def __neg__(self):
        return self * -1
    
    # Value - Value or Value - int/float
    def __sub__(self, other):
        return self + (-other)

    # int/float - Value
    def __rsub__(self, other):
        return self + (-other)

    # Value**int/float
    def __pow__(self, other):
        # Make sure that other is int/float
        assert isinstance(other, (int, float))

        out = Value(data=(self.data**other), _children=(self, ), _op=f"^{other:.3g}")

        l = f"({self.label})^{other:.3g}"
        self.set_new_label_creation(other)
        out.label = l if self.should_create_new_labels else f"{out.data:.3g}"

        def _backward():
            dout_dself = other * self.data**(other - 1)
            self.grad += out.grad * dout_dself
        out._backward = _backward

        return out

    # int/float**Value
    def __rpow__(self, other):
        return self**other
    
    # Value/Value or Value/(int/float)
    def __truediv__(self, other):
        return self * (other**-1)

    # (int/float)/Value
    def __rtruediv__(self, other):
        return self * (other**-1)

    def exp(self):
        out = Value(data=math.exp(self.data), _children=(self, ), _op="exp")

        l = f"exp({self.label})"
        self.set_new_label_creation(5)
        out.label = l if self.should_create_new_labels else f"{out.data:.3g}"

        def _backward():
            dout_dself = out.data
            self.grad += out.grad * dout_dself
        out._backward = _backward

        return out
        
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)   
    def tanh(self):
        out = Value(data=math.tanh(self.data), _children=(self, ), _op="tanh")

        l = f"tanh({self.label})"
        self.set_new_label_creation(6)
        out.label = l if self.should_create_new_labels else f"{out.data:.3g}"

        def _backward():
            dout_dself = 1 - out.data**2
            self.grad += out.grad * dout_dself
        out._backward = _backward

        return out





class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    # Every neuron has a number of weights and one bias
    def __init__(self, n_weights):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_weights)]
        self.bias = Value(random.uniform(-1, 1))
    
    # Returns the squashed sum of the bias with the dot product of the weights and inputs
    def __call__(self, inputs):
        activation = sum((weight_i * input_i for weight_i, input_i in zip(self.weights, inputs)), self.bias)
        out = activation.tanh()
        return out

    # Returns the list of weights and the bias of the neuron
    def parameters(self):
        return [self.bias] + self.weights

    # Cosmetic stuff
    def __repr__(self):
        name = ""
        w = self.weights[0]
        start = w.label.index("N")
        name = w.label[start:]
        return f"Neuron(w={[round(w.data, 3) for w in self.weights]}, b={round(self.bias.data, 3)}, name={name})"

class Layer(Module):
    # Every layer has a number of neurons, where each Neuron has the same number of weights as the other
    def __init__(self, n_neurons, n_weights_per_neuron):
        self.neurons = [Neuron(n_weights_per_neuron) for _ in range(n_neurons)]

    # Return the list of the layer's neurons evaluated for the same set of inputs
    def __call__(self, inputs):
        outs = [n(inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # Return all of the weights and biases of the layer's neurons
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    # Cosmetic stuff again
    def __repr__(self):
        return f"Layer({self.neurons})"

class MLP(Module):
    # Every MLP consists of layers. The number of weights for each neuron of layer is equal to the number of neurons of the previous layer. The number of weights of the neurons of the first layer equal the number of inputs
    def __init__(self, n_inputs, n_neurons_per_layer: list):
        size = [n_inputs] + n_neurons_per_layer
        self.layers = [Layer(n_neurons=size[i + 1], n_weights_per_neuron=size[i]) for i in range(len(n_neurons_per_layer))]

        # Just creating labels for each of the weights and biases 
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                for weight_index, weight in enumerate(neuron.weights):
                    weight.label = f"W{weight_index}-N{neuron_index}-L{layer_index}"
                neuron.bias.label = f"B-N{neuron_index}-L{layer_index}"

    # The input for any layer are the evaluated neurons of the previous layer
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    # Returns a list of all of the weights and biases of the MLP
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    # Just cosmetic stuff again
    def __repr__(self):
        ls = ""
        for l in self.layers:
            ls = ls + f"{l}\n"
        output = "MLP\n" + ls
        return output





def main():
    # Define the set of inputs and desired outputs
    inputs = [
        [Value(2.0, label="input_1"), Value(3.0, label="input_2"), Value(-1.0, label="input_3")],
        [Value(3.0, label="input_4"), Value(-1.0, label="input_5"), Value(0.5, label="input_6")],
        [Value(0.5, label="input_7"), Value(1.0, label="input_8"), Value(1.0, label="input_9")],
        [Value(1.0, label="input_10"), Value(1.0, label="input_11"), Value(-1.0, label="input_12")]
    ]
    desired_outputs = [Value(1.0, label="desired1"),
                    Value(-1.0, label="desired2"),
                    Value(-1.0, label="desired3"),
                    Value(1.0, label="desired4")
    ]

    # Define the MLP
    n_inputs = len(inputs[0])
    mlp = MLP(n_inputs, [4, 4, 1])


    n_interations = 1000
    # Training loop
    for k in range(n_interations):
        # Forward pass
        predictions = [mlp(input_) for input_ in inputs] 
        loss = sum(((desired_output - prediction)**2 for desired_output, prediction in zip(desired_outputs, predictions)))
        
        if k == 0:
            before = loss
        elif k == range(n_interations)[-1]:
            after = loss

        # Zero grad
        for p in mlp.parameters():
            p.grad = 0 

        # Backward pass
        loss.backward()

        # Gradient descent
        for p in mlp.parameters():
            p.data += (-0.01) * p.grad

        # Print the updated loss
        print(k, loss.data)

    print("Loss before:", before.data, "\nLoss after:", after.data)

    print("Updated predictions:")
    print([mlp(input_).data for input_ in inputs])

    after.create_graph()

if __name__ == "__main__":
    main()