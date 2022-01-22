import numpy as np

def _max_of_three(triple):
    l,m,r = triple
    if l<m and r<m:
        return True
    return False

def _min_of_three(triple):
    l,m,r = triple
    if l>m and r>m:
        return True
    return False

def estimate_circle_center(prediction_data, sample_start, sample_end):
    data = prediction[sample_start:sample_end]

    zx_C_maxset_alpha = []
    Cx_alpha_localmax_pos = []
    zx_C_minset_alpha = []
    Cx_alpha_localmin_pos = []

    zy_C_maxset_alpha = []
    Cy_alpha_localmax_pos=[]
    zy_C_minset_alpha = []
    Cy_alpha_localmin_pos=[]

    for i in range(1, len(data.shape[0]) - 1):
        x_l, y_l = data[i - 1,]
        x_m, y_m = data[i,]
        x_r, y_r = data[i + 1,]
        if _max_of_three(x_l, x_m, x_r):
            zx_c_maxset_alpha.append(data[i + 1])
            Cx_alpha_localmax_pos.append(sample_start + i + 1)
        elif _min_of_three(x_l, x_m, x_r):
            zx_C_minset_alpha.append(data[i + 1])
            Cx_alpha_localmin_pos.append(sample_start + i + 1)

        if _max_of_three(y_l, y_m, y_r):
            zy_C_maxset_alpha.append(data[i + 1])
            Cy_alpha_localmax_pos.append(sample_start + i + 1)
        elif _min_of_three(y_l, y_m, y_r):
            zy_C_minset_alpha.append(data[i + 1])
            Cy_alpha_localmin_pos.append(sample_start + i + 1)

    center = np.array([(np.average(zx_C_maxset_alpha) + np.average(zx_C_minset_alpha))/2, 
            (np.average(zy_C_maxset_alpha) + np.average(zy_C_minset_alpha))/2])

    return center, Cx_alpha_localmax_pos, Cy_alpha_localmax_pos, Cx_alpha_localmin_pos, Cy_alpha_localmin_pos

def direction_of_rotation_stricter(data, x_localmax_pos, y_localmax_pos, x_localmax_min, y_localmal_min, stepback):
    c_vel_x_max = []
    c_vel_x_min = []
    for i in x_localmax_pos:
        if data[i, 0] > 0:
            c_vel_x_max.append((data[i, 1] - data[i - stepback, 1]) / stepback)
        elif data[i, 0] < 0:
            c_vel_x_min.append((data[i, 1] - data[i - stepback, 1]) / stepback)

    c_vel_y_max = []
    c_vel_y_min = []
    for i in y_localmax_pos:
        if data[i, 1] > 0:
            c_vel_y_max.append((data[i, 0] - data[i - stepback, 0]) / stepback)
        elif data[i, 1] < 0:
            c_vel_y_min.append((data[i, 0] - data[i - stepback, 0]) / stepback)

    if all(i > 0 for i in c_vel_x_max) and all(i < 0 for i in c_vel_x_min) and all(i < 0 for i in c_vel_y_max) and all(i > 0 for i in c_vel_y_min):
        c_vel_dir = 1
    elif all(i < 0 for i in c_vel_x_max) and all(i > 0 for i in c_vel_x_min) and all(i > 0 for i in cel_vel_y_max) and all(i < 0 for i in cel_vel_y_min):
        c_vel_dir = -1
    else:
        c_vel_dir = 0
    
    return c_vel_dir, np.array(c_vel_x_max), np.array(c_vel_y_max), np.array(c_vel_x_min), np.array(c_vel_y_min)

def roundness(x, y, x_localmax_pos, circle_center):
    res = np.array(list(zip(localmax_pos, localmax_pos[1:])))
    test_dist_avg = []
    for i in range(len(res)):
        # What is this doing how is this working?
        test_dist=distance_2pts([x[res[i][0]:res[i][1]],y[res[i][0]:res[i][1]]], circle_center)
        roundness=np.amax(test_dist)-np.amin(test_dist)
        test_dist_avg.append(roundness)

    return np.average(test_dist_avg)

def list_to_check_if_LC(x, sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol):
    z_C_Wout_alpha = x[sample_start:sample_end]
    z_C_Wout_alpha_set = []

    for i in range(1, len(z_C_Wout_alpha)):
        if _max_of_three(z_C_Wout_alpha[i-1:i+2]):
            z_C_Wout_alpha_set.append(z_C_Wout_alpha[i+1])

    Cklist = []
    if len(z_C_Wout_alpha_set) != 0:
        pass
    # for some reason only y coordinate
    raise NotImplementedError("WTF")

def list_to_check_if_LC_v3(x, sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, round_no):









