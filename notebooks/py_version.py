import numpy as np


COL_STEPS = 20
PLUME_HEIGHT = 24000
VENT_ELEVATION = 3085
ALPHA = 1.02
BETA = 1.56


def plume_pdf2(x_norm, step, alpha, beta, sum_prob):

    i = 0
    x = x_norm
    a1 = alpha - 1.0
    b1 = beta - 1.0
    probability = 0.0
    if not sum_prob:
        for i in range(COL_STEPS):
            x += step
            if (x <= 0):
                # step is the small slice of the column as a fraction of the
                # whole
                x1 = x + 0.001
                x2 = 1.0 - x1
                prob = (x1 ** a1) * (x2 ** b1)
            elif (x >= 1):
                x1 = x - 0.001
                x2 = 1.0 - x1
                prob = (x1 ** a1) * (x2 ** b1)
            else:
                x1 = 1.0 - x
                prob = (x ** a1) * (x1 ** b1)
            print("[sum_prob=0][%d] x=%g step=%g prob=%g" % (i, x, step, prob))
            probability += prob

    # Just calculate the probability for one column step
    else:
        x1 = 1.0 - x
        probability = (x ** a1) * (x1 ** b1)
    if (np.isnan(probability)):
        probability = 0
    return probability


ht_section_width = PLUME_HEIGHT - VENT_ELEVATION
ht_step_width = ht_section_width / COL_STEPS
step_norm = ht_step_width / ht_section_width
x_norm = 0.0
total_P_col = 0.0
print(x_norm, step_norm, ALPHA, BETA, total_P_col)
cum_prob_col = plume_pdf2(x_norm, step_norm, ALPHA, BETA, total_P_col)

