# toygrad

A toy forward-mode autodiff utility written in Python.

![](https://media.giphy.com/media/13pT5ZMDTKiKJO/giphy.gif)

## Usage

```python
from toygrad import grad

def f(x, y):
    return (x * x * 3.0 + y * y * y) * 2.0

print(grad(f, x=1.0, y=2.0)) #=> {'x': 12.0, 'y': 24.0}
```
