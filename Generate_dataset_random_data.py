import random
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy.integrate import solve_ivp
import csv
import pandas as pd
import os
import cvxpy as cp
import signal
from scipy.optimize import minimize, NonlinearConstraint

N_CONSTRAINTS = 10
M_DIM = 10
save_dir = '.'
os.makedirs(save_dir, exist_ok=True)

howmanysamples = 50000

# General Dimension 
def sample_in_unit_ball(dim):
    """
    Uniform sample from the dim-dimensional unit ball.
    """
    v = np.random.normal(size=dim)
    norm = np.linalg.norm(v)
    if norm < 1e-10: return np.zeros(dim)
    v = v / norm
    r = np.random.uniform(0, 1) ** (1/dim)
    return r * v


# MIN NORM QP FOR INITIAL GUESS (General N, m) 
def min_norm_controller(A_list, B_list):
    """
    Solve:
      minimize ||u||^2
      subject to: A_i + B_i^T u <= -0.05 for i=1..N
    where u ∈ R^m
    """
    u = cp.Variable(M_DIM)
    
    constraints = [
        A_list[i] + B_list[i].T @ u <= -0.05 
        for i in range(N_CONSTRAINTS)
    ]

    objective = cp.Minimize(0.5 * cp.sum_squares(u))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()
        return u.value
    except:
        return None


# COST FUNCTION AND GRADIENT 
def J_p(k, A_list, B_list, r):
    total_cost = 0
    k_sq_norm = np.linalg.norm(k)**2
    
    for i in range(N_CONSTRAINTS):
        denom = A_list[i] + B_list[i].T @ k
        if denom >= 0:
            return 1e12 # Penalty for infeasibility
        
        num = np.linalg.norm(B_list[i])**2 + r * k_sq_norm
        total_cost += num / denom
        
    return -0.5 * total_cost


def grad_J_p(k, A_list, B_list, r):
    grad = np.zeros_like(k)
    k_sq_norm = np.linalg.norm(k)**2
    
    for i in range(N_CONSTRAINTS):
        denom = A_list[i] + B_list[i].T @ k
        if denom >= 0:
            # Return a large gradient pointing inward
            return 1e8 * B_list[i] / (np.linalg.norm(B_list[i]) + 1e-10)
        
        num = np.linalg.norm(B_list[i])**2 + r * k_sq_norm
        grad += 0.5 * num * (B_list[i] / denom**2) - (r * k) / denom
        
    return grad


# OPTIMIZATION ROUTINE
def robust_minimize_Jp(A_list, B_list, r, k0):
    # 1. Define the constraints: A_i + B_i^T * k <= -1e-4
    # We write them as: -1e-4 - (A_i + B_i^T * k) >= 0
    cons = []
    for i in range(N_CONSTRAINTS):
        cons.append({
            'type': 'ineq', 
            'fun': lambda k, i=i: -1e-4 - (A_list[i] + np.dot(B_list[i], k))
        })

    res = minimize(
        fun=J_p,
        x0=k0,
        jac=grad_J_p,
        args=(A_list, B_list, r),
        method='SLSQP',
        constraints=cons,
        options={
            'ftol': 1e-12,  
            'disp': False,
            'maxiter': 1000,
        }
    )
    
    grad_val = grad_J_p(res.x, A_list, B_list, r)
    grad_norm = np.linalg.norm(grad_val)
    
    # Tight feasibility check
    is_feasible = all((A_list[i] + B_list[i].T @ res.x < -1e-5) for i in range(N_CONSTRAINTS))
    
    # Accuracy check: Success + Feasible + Converged Gradient
    is_accurate = res.success and is_feasible and (grad_norm < 1e-6)
    
    return res.x, is_accurate



# DATASET GENERATION
def generate_dataset(n=howmanysamples, lower=-1, upper=1):
    dataset = []
    attempts = 0
    
    while len(dataset) < n:
        attempts += 1
        
        # Sample A_i (scalars) and B_i (m-dim vectors)
        A_vals = [random.uniform(lower, upper) for _ in range(N_CONSTRAINTS)]
        B_vals = [sample_in_unit_ball(M_DIM) for _ in range(N_CONSTRAINTS)]
        r_val = random.uniform(0.0001, 1)

        # Warmstart QP
        x0 = min_norm_controller(A_vals, B_vals)
        if x0 is None or np.any(np.isnan(x0)):
            continue

        try:
            # Inside generate_dataset loop:
            kf, success = robust_minimize_Jp(A_vals, B_vals, r_val, x0.flatten())
            
            if success:
                final_grad = grad_J_p(kf, A_vals, B_vals, r_val)
                if np.linalg.norm(final_grad) < 1e-6:
                    dataset.append((A_vals, B_vals, r_val, kf))
                
                    if len(dataset) % 100 == 0:
                        print(f"Collected {len(dataset)} samples (Attempt {attempts})")

        except Exception as err:
            continue

    return dataset


# MAIN
def main():
    print(f"Generating data for N = {N_CONSTRAINTS}, m = {M_DIM} ...")
    data = generate_dataset(howmanysamples)

    full_path = '.'

    # Flatten the structures for CSV saving
    records = []
    for (A_list, B_list, r, kf) in data:
        row = {}
        for i in range(N_CONSTRAINTS):
            row[f"a{i}"] = A_list[i]
            for j in range(M_DIM):
                row[f"b{i}_{j}"] = B_list[i][j]
        row["r"] = r
        for j in range(M_DIM):
            row[f"kf_{j}"] = kf[j]
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(full_path, index=False)
    print(f" Dataset saved to: {full_path}")

    X_list = []
    y_list = []
    
    for (A_list, B_list, r, kf) in data:
        x_vec = []
        x_vec.extend(A_list)
        for b in B_list:
            x_vec.extend(b.tolist())
        x_vec.append(r)
        
        X_list.append(x_vec)
        y_list.append(kf.tolist())

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print("Input shape:", X.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()