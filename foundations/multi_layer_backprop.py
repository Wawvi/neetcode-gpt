import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        def forward_linear(x, w, b) :
            return np.dot(x, np.transpose(w)) + b

        def forward_relu(z) : 
            return np.maximum(0, z)
        
        def mse_loss(y_hat, y_true) :
            return np.mean((y_hat - y_true) ** 2)

        def loss_grad(y_hat, y_true) :
            return 2 * (y_hat - y_true) / len(y_true)

        def w2_grad(dz2, a1) :
            return np.outer(np.transpose(dz2), a1)
            
        def b2_grad(dz2) :
            return dz2
        
        def da1_grad(dz2, W2) :
            return np.outer(np.transpose(dz2), W2)

        def dz1_grad(da1, z1) :
            return np.where(z1 <= 0, 0, da1)
        
        def w1_grad(dz1, x) :
            return np.outer(np.transpose(dz1), x)

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = forward_linear(x, W1, b1)
        a1 = forward_relu(z1)
        z2 = forward_linear(a1, W2, b2)
        loss = mse_loss(z2, y_true)
        dz2 = loss_grad(z2, y_true)
        dw2 = w2_grad(dz2, a1)
        db2 = b2_grad(dz2)
        da1 = da1_grad(dz2, W2)
        dz1 = dz1_grad(da1, z1)
        dw1 = w1_grad(dz1, x)
        db1 = dz1

        return {
            "loss" : np.round(loss, 4),
            "dW1" : (np.round(dw1, 4) + 0.).tolist(),
            "db1" : np.round(db1, 4).tolist()[0] ,
            "dW2" : np.round(dw2, 4).tolist(),
            "db2" : np.round(db2, 4).tolist()
        }

