from toygrad import grad, dual

assert grad(lambda x: x + dual(3.0), x=3.0)["x"] == 1.0
assert grad(lambda x: x + 3.0, x=3.0)["x"] == 1.0
assert grad(lambda x: x ** 2.0, x=3.0)["x"] == 6.0
assert grad(lambda x: x ** dual(2.0), x=3.0)["x"] == 6.0
assert grad(lambda x: (x ** 2.0) * x, x=3.0)["x"] == 27.0
assert grad(lambda x, y: x ** 2 + y ** 2, x=3.0)["x"] == 6.0
assert grad(lambda x, y: x ** 2 + y ** 2, y=3.0)["y"] == 6.0
assert grad(lambda x, y: x ** 2 + y ** 2, x=3.0, y=4.0) == {"x": 6.0, "y": 8.0}

print("All assertions passed! ✨ 🍰 ✨")
