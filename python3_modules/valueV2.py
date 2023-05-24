import math
from graphviz import Digraph

class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def backward(self):
        
        topo_sorted = []
        seen = set()

        def explore(node):
            if node not in seen:
                seen.add(node)
                for child in node._prev:
                    explore(child)
                topo_sorted.append(node)

        # This changes the topo_sorted list to be a topological sort of our nodes 
        explore(self)

        self.grad = 1
        for node in reversed(topo_sorted):
            node._backward()

    def create_graph(self):

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
        
        def graph(root):
            nodes, edges = trace(root)

            dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}, filename="my_graph")

            for node in nodes:
                node_id = str(id(node))
                dot.node(name=node_id, label=f"{node.label} | data {node.data:.3f} | grad {node.grad:.3f}", shape="record")

                if node._op:
                    dot.node(name=(node_id+ node._op), label=node._op)
                    dot.edge((node_id + node._op), node_id)

            for child, parent in edges:
                dot.edge(str(id(child)), (str(id(parent)) + parent._op))

            return dot
    
        g = graph(self)
        g.render(directory="graphviz_outputs")

    def __repr__(self):
        return f"Value(data={self.data:.3f})" if self.label=="" else f"Value(\"{self.label}\", data={self.data:.3f})"

    def __add__(self, other):
        
        other = other if isinstance(other, Value) else Value(data=other, label=f"{other}")

        out = Value(data=(self.data + other.data), _children=(self, other), _op="+")

        l = f"{self.label} + {other.label}" if (self.label != "" and other.label != "") else ""
        out.label = l 

        def _backward():
            dout_dself = 1
            dout_dother = 1

            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(data=other, label=f"{other}")

        out = Value(data=(self.data * other.data), _children=(self, other), _op="*")
        
        l = f"({self.label})*{other.label}" if (self.label != "" and other.label != "") else ""
        out.label = l 

        def _backward():
            dout_dself = other.data
            dout_dother = self.data

            self.grad += out.grad * dout_dself
            other.grad += out.grad * dout_dother
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):

        assert isinstance(other, (int, float))

        out = Value(data=(self.data**other), _children=(self, ), _op=f"^{other}")

        l = f"({self.label})^{other}" if not (self.label == "") else ""
        out.label = l

        def _backward():
            dout_dself = other * self.data**(other - 1)
            self.grad += out.grad * dout_dself
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def raise_euler(self):
        out = Value(data=math.exp(self.data), _children=(self, ), _op=f"e^{self.data}")

        l = f"e^({self.label})" if not (self.label == "") else ""
        out.label = l

        def _backward():
            dout_dself = math.exp(self.data)
            self.grad += out.grad * dout_dself
        out._backward = _backward

        return out

def test0():
    # Define the initial expression
    a = Value(2.0, label="")
    b = Value(3.0, label="b")
    c = a + b
    d = Value(-1.0, label="d")
    e = c * d
    f = 2 * e
    g = 10 + f
    h = g + g
    i = h + 4
    j = i**3
    k = i / j
    two_k = 2*k
    e_power_two_k = two_k.raise_euler()
    e_power_two_k_minus_one = e_power_two_k - 1
    e_power_two_k_plus_one = e_power_two_k + 1
    tanh_k = e_power_two_k_minus_one / e_power_two_k_plus_one 
    tanh_k.create_graph()
    
    for _ in range(10):
        # Reset the gradients
        tanh_k.grad = 0
        e_power_two_k_plus_one.grad = 0
        e_power_two_k_minus_one.grad = 0
        e_power_two_k.grad = 0
        two_k.grad = 0
        k.grad = 0
        i.grad = 0
        h.grad = 0
        g.grad = 0
        f.grad = 0
        e.grad = 0
        d.grad = 0
        c.grad = 0
        b.grad = 0
        a.grad = 0

        # Perform the backward pass
        tanh_k.backward()

        print("Before decreasing:", tanh_k.data)

        # Decrease the final output k
        a.data += (-0.1) * a.grad

        # Perform forward pass
        c = a + b
        d = Value(-1.0, label="d")
        e = c * d
        f = 2 * e
        g = 10 + f
        h = g + g
        i = h + 4
        j = i**3
        k = i / j
        two_k = 2*k
        e_power_two_k = two_k.raise_euler()
        e_power_two_k_minus_one = e_power_two_k - 1
        e_power_two_k_plus_one = e_power_two_k + 1
        tanh_k = e_power_two_k_minus_one / e_power_two_k_plus_one 
        
        print("After decreasing:", tanh_k.data)

        print()

if __name__ == "__main__":
    test0()