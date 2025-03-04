import numpy as np

#the weights before updated
w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.3
w5, w6, w7, w8 = 0.4, 0.45, 0.5, 0.55

# Learning rate
eta = 0.5  

# Error percentage regarding weight
dEtotal_dw1 = 0.00044  
dEtotal_dw2 = 0.00087  
dEtotal_dw3 = 0.00049  
dEtotal_dw4 = 0.00099  
dEtotal_dw5 = 0.0822  
dEtotal_dw6 = 0.0822  
dEtotal_dw7 = -0.0277  
dEtotal_dw8 = -0.0277  

# updated weights
w1_new = w1 - eta * dEtotal_dw1
w2_new = w2 - eta * dEtotal_dw2
w3_new = w3 - eta * dEtotal_dw3
w4_new = w4 - eta * dEtotal_dw4
w5_new = w5 - eta * dEtotal_dw5
w6_new = w6 - eta * dEtotal_dw6
w7_new = w7 - eta * dEtotal_dw7
w8_new = w8 - eta * dEtotal_dw8


print(f"Updated Weights:")
print(f"w1 = {w1_new:.5f}")
print(f"w2 = {w2_new:.5f}")
print(f"w3 = {w3_new:.5f}")
print(f"w4 = {w4_new:.5f}")
print(f"w5 = {w5_new:.5f}")
print(f"w6 = {w6_new:.5f}")
print(f"w7 = {w7_new:.5f}")
print(f"w8 = {w8_new:.5f}")