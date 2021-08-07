# toygrad

A toy forward-mode autodiff utility written in Python.

![](https://media.giphy.com/media/13pT5ZMDTKiKJO/giphy.gif)

## Usage

```python
from toygrad import gradient

# Compute the partial derivative ∂/∂x = 2x / y^2 of `func`
# and evaluate the derivative at x = 2 and y = 3.
func = lambda x, y: (x ** 2) / (y ** 2)
print(gradient(of=func, wrt="x", at=[2, 3]))
```
