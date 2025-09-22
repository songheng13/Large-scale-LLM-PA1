# PA1 Writeup

## 1.5

### Testing Method
To test the implementation of automatic differentation given scalar inputs, I compared the results of symbolic, numercial, and automatic differentiation on functions f_1, f_2 and f_3 on the inputs xs = [0.5, 1.0, 2.5]. I also compared the results of numerical and automatic differentiation on function f_4 on the same scalar inputs xs = [0.5, 1.0, 2.5]. 

Specifically for symbolic differentiation, I used the previously implemented derivatives df1dx, df2dx and df3dx for functions f_1, f_2 and f_3, respectively. For getting the numerical derivative, I ran the given method numerical_diff.

### Results

===== At x = 0.5 =====
|            | symbolic | numerical | automatic |
| ---------- | -------- | --------- | --------- | 
| df_1/dx | 2 | 2.0000000000131024 | 2.0 |
| df_2/dx | 1.0 | 0.9999999999982244 | 1.0 |
| df_3/dx |  -0.11834319526627218 | -0.11834319526005109 | -0.11834319526627218 |
| df_4/dx |  | -2.804687464652566 | -2.8046874646640316 |

===== At x = 1.0 =====

|            | symbolic | numerical | automatic |
| ---------- | -------- | --------- | --------- | 
| df_1/dx | 2 | 2.0000000000131024 | 2.0 |
| df_2/dx | 2.0 | 2.000000000002 | 2.0 |
| df_3/dx | 0.0  | 2.775557561562891e-11 | 0.0 |
| df_4/dx |  | -2.3499102726587395 | -2.3499102727038075

===== At x = 2.5 =====

|            | symbolic | numerical | automatic |
| ---------- | -------- | --------- | --------- | 
| df_1/dx | 2 | 1.9999999999686933 | 2.0 |
| df_2/dx | 5.0 | 5.000000000032756  | 5.0 |
| df_3/dx | 0.48 | 0.4800000000193538 | 0.48
| df_4/dx|  | 0.214305259155223 | 0.2143052591764773

### Observations
Given some $x$ in test inputs $xs$, I observed that while the results of numerical differentiation were slightly off by < (10e-8) from the outputs of symbolic and automatic differentiation, symbolic and automatic differentiation results were exactly identical. Since this difference is negligible, I concluded that the implementation of backpropogation and automatic differentiation is correct for scalar inputs.


## 2.3

### Problem

The starter implementations of BA_Add.grad_fn (and similarly Sub, Mul, Div) assume that inputs x and y always have the same shape or are scalars. This fails when broadcasting occurs (e.g., x.shape = (3,1) and y.shape = (1,4)), because the upstream gradient will have a larger shape than the original inputs. Directly adding self.grad leads to shape mismatch errors.

### Fix

We introduced an unbroadcast helper that reduces the gradient back to the original shape by summing along broadcasted dimensions. Each grad_fn now uses this helper to correctly align the gradient with the shape of the original inputs.

## 2.4

### Testing Method

We compared numerical differentiation (central difference with eps=1e-5) and automatic differentiation (backpropagation) on two functions, g1 and g2. Tests were performed at x = 0.5.

### Results

=== Testing g1 ===
g1: numerical = 9.000000000014552  autodiff = 9.0

=== Testing g2 ===
g2: numerical = 377.79462820850534  autodiff = 377.79462821241566

### Analysis

- For g1, both methods give ~9, with a negligible difference (~1e-11).
- For g2, both methods give ~377.79, with a negligible difference (~1e-9).
- These small discrepancies are due to floating-point precision.

### Conclusion

Automatic differentiation matches numerical differentiation within floating-point error, confirming the correctness of the implementation.

## 2.6

### Testing Method

To perform the test, we construct an array [1.0, 2.0, 3.0, 4.0, 5.0] and compute its gradient under both the numerical gradient function (with epsilon equals to 1e-5) and automatic differentiation.

### Results

=== numerical_diff ===
numerical: [0. -4. 8. 48. 127.99999999]
=== backprop_diff ===
backprop: [0. -4. 8. 48. 128.]

### Analysis

From the results, we observe that, except for the last value in the array, all other values in numerical differentiation and backpropagation are identical. The difference in the last value between these two implementations is very small (1e-08), which confirms the correctness of our numerical implementation.

## 2.7
From the results, we find that the time elapsed for automatic differentiation is around 0.09 ms, whereas for numerical differentiation it is around 11.16 ms, which is more than 100 times longer. Therefore, the backprop differentiation is much faster than numerical differentiation.

## Summary
While implementing automatic differentiation for function that take and in and output both scalars and multi-dimensional values, we noticed that differentiations in scalars is much more straight forward than for multi-dimensional.
For multi-dimensional values, we also need to guarantee that the input and output dimensions after calculating gradients remain the same. For scalar differentiation, we know that the derivative output will also be a scalar. However, after calculating gradients given multi-dimensional inputs, we need to broadcast the output back to the input dimensions.

Given arbitrary dimensional inputs and functions, we noticed that when comparing the results of symbolic, numerical, and automatic differentiation during testing, that symbolic and automatic differentiation would have identical results, while numerical differentiation sees very negligble difference. This is because, given some function f, numerical differentiation is an estimate of the actual gradient df/dx. Specifically, the first order Taylor approximation: $$ \frac{df}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h},$$ assuming local linearity when pertubation $h \rightarrow 0$. On the other hand, symbolic and automatic differentiation are the actual results of df/dx.
