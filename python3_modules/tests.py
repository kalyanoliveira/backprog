# Intended to be used only on final.py, where all of the dependencies are already imported

def test1():
    Value.should_show_labels = True
    a = Value(2.0, label="a")
    b = Value(-1.0, label="b")
    c = a * b
    d = 5 + c
    e = 2 * d
    print(a, b, c, d, e)
    f = e + e
    g = Value(-1, label="g")
    h = f * g
    i = 4 - h
    j = 2**i 
    k = 4/j
    l = k.exp()
    m = l.tanh()
    m.backward()
    m.create_graph()

def test2():
    a = Value(0.7)
    b = Value(-0.85)
    c = a * b
    print(c.label)
    d = c.tanh()
    d.backward()
    d.create_graph()

def test3():
    inputs = [
        [Value(2.0, label="1"), Value(3.0, label="2"), Value(-1.0, label="3")],
        [Value(3.0, label="4"), Value(-1.0, label="5"), Value(0.5, label="6")],
        [Value(0.5, label="7"), Value(1.0, label="8"), Value(1.0, label="9")],
        [Value(1.0, label="10"), Value(1.0, label="11"), Value(-1.0, label="12")],
    ]
    n = Neuron(len(inputs))
    out = n(inputs[0])
    out.create_graph()

def test4():
    inputs = [1, 2, 3]
    mlp = MLP(len(inputs), [4, 4, 1])
    out = mlp(inputs)
    print(out)
    out.create_graph()