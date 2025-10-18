import numpy as np


# Perform linear regression on the given cost dictionary within the specified bounds [lb, ub]
# truncate_val is used to scale down the slope and intercept
def linear_regression(cost_dict, lb, ub, truncate_val=1):
    x = []
    y = []
    for i in range(lb, ub+1):
        assert i in cost_dict.keys()
        x.append(i)
        y.append(cost_dict[i])
    slope, intercept = np.polyfit(x, y, 1)
    return (round(slope/truncate_val), round(intercept/truncate_val))

def linear_regression_max(cost_dict, lb, ub, truncate_val=1):
    max_val = max(cost_dict[i] for i in range(lb, ub+1))
    return (0, round(max_val/truncate_val))