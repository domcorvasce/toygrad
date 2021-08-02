# toygrad

A toy forward-mode autodiff utility written in Python.

## Usage

```python
from toygrad import grad, dual, exp

def f(x, y):
    return (x * x * 3 + y * y * y) * 2

print(grad(f, x=1.0, y=2.0)) #=> {'x': 12.0, 'y': 24.0}
```
