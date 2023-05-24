import math
from graphviz import Digraph

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None



    # Operations
    # Value + Value
    def __add__(self, other):
        # If other is a Value object, keep it
        # Else (if other is just a int/float, for instance), create a Value object based on other
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")
    
        # Create out's _backward behavior
        def _backward():
            # Compute the local derivatives of out with respect to self and other
            # and change self's and other's grads
            dout_dself = 1.0
            dout_dother = 1.0
            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother

        # Set out's _backward behavior to what it should be
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    # Value * Value
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            dout_dself = other.data
            dout_dother = self.data
            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother

        out._backward = _backward

        return out

    # -Value
    def __neg__(self):
        # self.__mul__(-1)
        return self * -1
    
    # Value - Value
    def __sub__(self, other):
        # self.__add__(other.__neg__)
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)

    # If python cannot do other * self (for instance, when "other" is an int/float),
    # then this function gets called
    # as in 2 * Value
    def __rmul__(self, other):
        return self * other

    # (Value)^int
    def __pow__(self, other):
        # Make sure that other is a float or int
        assert isinstance(other, (float, int))
        
        out = Value(self.data**other, (self, ), f"^{other}")

        def _backward():
            dout_dself = other * self.data**(other - 1)
            self.grad += out.grad * dout_dself

        out._backward = _backward

        return out

    # Value / Value
    def __truediv__(self, other):
        # self.__mul__(other.__pow__(-1))
        return self * other**-1
    
    def __rtruediv__(self, other):
        return self * other**-1

    # e^(Value)
    def exp(self):
        out = Value(math.exp(self.data), (self, ), "e^")
        
        def _backward():
            dout_dself = out.data
            self.grad += out.grad * dout_dself

        out._backward = _backward

        return out

    # tanh(Value)
    def tanh(self):
        out = Value(math.tanh(self.data), (self, ) , "tanh")

        def _backward():
            dout_dself = 1 - out.data**2
            self.grad += out.grad * dout_dself

        out._backward = _backward

        return out



    # Backprog stuff
    def backward(self):
        # Treat self as a root node
        # dself/dself = 1.0, hence
        self.grad = 1.0

        # Create a list of the sorted nodes of a digraph
        # where the digraph is simply 
        # the graph of the expression of the Value objects leading up to self
        topo_sorted = []
        seen = set()

        def explore(node):
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    explore(child)
                topo_sorted.append(node)

        # This updates our "topo_sorted" and "seen" local variables

        explore(self) 

        # For every node in our reversed, topologically-sorted digraph,
        # call its _backward behavior.
        # reversed because we want to call the leaf nodes last,
        # and the leaf nodes are the nodes with least in-degrees (topo-sort jargon) 
        for node in reversed(topo_sorted): 
            node._backward()



    # Cosmetic stuff
    def __repr__(self):
        return f"Value(\"{self.label}\", data={self.data})"

    def create_graph(self):
        # Create a set of known edges (connections between Value objects) 
        # and known nodes (known Value objects)
        def trace(root):
            nodes, edges = set(), set()

            def build(node):
                if node not in nodes:
                    nodes.add(node)
                    for child in node._prev:
                        edges.add((child, node))
                        build(child)

            build(root)
            return nodes, edges

        def draw_dot(root):
            nodes, edges = trace(root)

            dot = Digraph(format="svg")

            # For each Value object, create a node for it
            for node in nodes:
                dot.node(name=str(id(node)), label=f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f}", shape="record")

                # If this node was created by an operation, we must create a node for that too
                # and then connect that operation node to the Value object node
                if node._op:
                    dot.node(name=str(id(node)) + node._op, label=node._op)
                    dot.edge(str(id(node)) + node._op, str(id(node)))

            # For each known connection between two Value objects
            for child, parent in edges:
                # Connect the child to the parent's operation node
                dot.edge(str(id(child)), str(id(parent))+parent._op)

            return dot

        graph = draw_dot(self)
        graph.render(directory="graphviz_outs")

def main():
    # Just building out the expression for a Neuron
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    x1w1 = x1*w1; x1w1.label="x1*w1"
    x2w2 = x2*w2; x2w2.label="x2*w2"

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"

    b = Value(6.8813735870195432, label="b")

    n = x1w1x2w2 + b; n.label="n"

    o = n.tanh(); o.label="o"
    
    # Backpropagate through the Neuron, and display its expression graph
    o.backward()
    o.create_graph()

def main2():
    a = Value(3.0)
    b = Value(2.0)

    print(a / b)

def main3():
    # Just building out the expression for a Neuron
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    x1w1 = x1*w1; x1w1.label="x1*w1"
    x2w2 = x2*w2; x2w2.label="x2*w2"

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"

    b = Value(6.8813735870195432, label="b")

    n = x1w1x2w2 + b; n.label="n"
    
    two_n = 2*n; two_n.label="2*n"

    e = two_n.exp(); e.label = "e^(2*n)"

    e_minus_1 = e - 1; e_minus_1.label = "e - 1"

    e_plus_1 = e + 1; e_plus_1.label = "e + 1"

    o = e_minus_1 / e_plus_1; o.label = "o"

    o.backward()
    o.create_graph()
    
if __name__ == "__main__":
    main3()
