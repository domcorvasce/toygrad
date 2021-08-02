# toygrad

A toy forward-mode autodiff utility written in Python.

## Usage

```python
from toygrad import grad, dual, exp

def f(x, y, z):
    return (x + y + z) * (exp ** (x * y * z))

print(grad(f, x=1.0, y=2.0, z=3.0)) #=> {'x': 2.0, 'y': 3.0, 'z': 4.0}
```
