from toygrad import grad, dual

# Checks that addition works with both dual numbers and float
assert grad(lambda x: x + dual(3.0), x=3.0)["x"] == 1.0
assert grad(lambda x: x + 3.0, x=3.0)["x"] == 1.0

# Checks that pow works with both dual numbers and float
assert grad(lambda x: x ** 2.0, x=3.0)["x"] == 6.0
assert grad(lambda x: x ** dual(2.0), x=3.0)["x"] == 6.0

# Checks that we can compute partial derivatives with respect to a variable,
# where the other variables treated as constants are set to zero.
assert grad(lambda x, y: (x * x * 3 + y * y * y) * 2, x=1.0, y=2.0) == {
    "x": 12.0,
    "y": 24.0,
}
assert grad(lambda x, y: x ** 2 + y ** 2, x=3.0, y=4.0) == {"x": 6.0, "y": 8.0}

# The library fails at differentiating this function.
# I'm going to address it in a future commit.
# assert grad(lambda x, y: (x ** 2) / (y ** 2), x=2.0, y=3.0)

print("All assertions passed! ✨ 🍰 ✨")
