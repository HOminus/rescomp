import numpy as np

def _max_of_three(l, m, r):
    if l<m and r<m:
        return True
    return False

def _min_of_three(l, m, r):
    if l>m and r>m:
        return True
    return False

def estimate_circle_center(prediction_data, sample_start, sample_end):
    if sample_start == None and sample_end == None:
        data = prediction_data
    else:
        data = prediction_data[sample_start:sample_end]

    zx_C_maxset_alpha = []
    Cx_alpha_localmax_pos = []
    zx_C_minset_alpha = []
    Cx_alpha_localmin_pos = []

    zy_C_maxset_alpha = []
    Cy_alpha_localmax_pos=[]
    zy_C_minset_alpha = []
    Cy_alpha_localmin_pos=[]

    for i in range(1, data.shape[0] - 1):
        x_l, y_l = data[i - 1,]
        x_m, y_m = data[i,]
        x_r, y_r = data[i + 1,]
        if _max_of_three(x_l, x_m, x_r):
            zx_C_maxset_alpha.append(data[i, 0])
            Cx_alpha_localmax_pos.append(sample_start + i)
        elif _min_of_three(x_l, x_m, x_r):
            zx_C_minset_alpha.append(data[i, 0])
            Cx_alpha_localmin_pos.append(sample_start + i)

        if _max_of_three(y_l, y_m, y_r):
            zy_C_maxset_alpha.append(data[i, 1])
            Cy_alpha_localmax_pos.append(sample_start + i)
        elif _min_of_three(y_l, y_m, y_r):
            zy_C_minset_alpha.append(data[i, 1])
            Cy_alpha_localmin_pos.append(sample_start + i)

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
    elif all(i < 0 for i in c_vel_x_max) and all(i > 0 for i in c_vel_x_min) and all(i > 0 for i in c_vel_y_max) and all(i < 0 for i in c_vel_y_min):
        c_vel_dir = -1
    else:
        c_vel_dir = 0
    
    return c_vel_dir#, np.array(c_vel_x_max), np.array(c_vel_y_max), np.array(c_vel_x_min), np.array(c_vel_y_min)

def roundness(data, x_localmax_pos, circle_center):
    test_dist_avg = []
    for i in range(len(x_localmax_pos) - 1):
        one_period = data[x_localmax_pos[i]:x_localmax_pos[i + 1],]
        distances = np.linalg.norm(one_period - circle_center, axis = 1)
        roundness=np.amax(distances) - np.amin(distances)
        test_dist_avg.append(roundness)
    return np.average(test_dist_avg)

def list_to_check_if_LC(y, sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol):
    z_C_Wout_alpha = y[sample_start:sample_end]
    z_C_Wout_alpha_set = []

    for i in range(1, len(z_C_Wout_alpha) - 1):
        l = z_C_Wout_alpha[i - 1]
        m = z_C_Wout_alpha[i]
        r = z_C_Wout_alpha[i + 1]
        if _max_of_three(l, m, r):
            z_C_Wout_alpha_set.append(z_C_Wout_alpha[i])

    Cklist = []
    if len(z_C_Wout_alpha_set) != 0 and all(i < FP_err_lim for i in abs(np.diff(y[FP_sample_start:FP_sample_end:50])))==False:
        zmax1 = np.amax(z_C_Wout_alpha_set)
        for i in range(len(z_C_Wout_alpha_set)):
            ztest1 = z_C_Wout_alpha_set[i]
            if abs((ztest1 - zmax1)/zmax1) <= LC_err_tol:
                Cklist.append(i)
    else:   
        Cklist.append(0.0)
        Cklist.append(10.0)
        Cklist.append(100.0)
        Cklist.append(1000.0)
    Ckdifflist=np.diff(Cklist)
    return Cklist,Ckdifflist

def list_to_check_if_LC_v3(x, sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no):
    z_C_Wout_alpha = x[sample_start:sample_end]
    z_C_Wout_alpha_maxset = []
    z_C_Wout_alpha_minset = []
    for i in range(1, len(z_C_Wout_alpha) - 1):
        l = z_C_Wout_alpha[i - 1]
        m = z_C_Wout_alpha[i]
        r = z_C_Wout_alpha[i + 1]
        if _max_of_three(l, m, r):
            z_C_Wout_alpha_maxset.append(z_C_Wout_alpha[i])
        elif _min_of_three(l, m, r):
            z_C_Wout_alpha_minset.append(z_C_Wout_alpha[i])

    C_maxarg_list = []
    C_minarg_list = []
    if len(z_C_Wout_alpha_maxset) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        zmax1 = np.amax(z_C_Wout_alpha_maxset)
        for i in range(len(z_C_Wout_alpha_maxset)):
            ztest1 = z_C_Wout_alpha_maxset[i]
            if abs((ztest1 - zmax1)/zmax1) <= LC_err_tol:
                C_maxarg_list.append(i)
    else:
        C_maxarg_list.append(0.0)
        C_maxarg_list.append(10.0)
        C_maxarg_list.append(100.0)
        C_maxarg_list.append(1000.0)
    C_maxarg_difflist=np.diff(C_maxarg_list)

    if len(z_C_Wout_alpha_minset) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        zmin1 = np.amin(z_C_Wout_alpha_minset)
        for i in range(len(z_C_Wout_alpha_minset)):
            ztest1 = z_C_Wout_alpha_minset[i]
            if abs((ztest1 - zmin1)/zmin1) <= LC_err_tol:
                C_minarg_list.append(i)
    else:
        C_minarg_list.append(0.0)
        C_minarg_list.append(10.0)
        C_minarg_list.append(100.0)
        C_minarg_list.append(1000.0)
    C_minarg_difflist=np.diff(C_minarg_list)    
    
    if len(C_maxarg_difflist) >= 2 and all(j == C_maxarg_difflist[0] for j in C_maxarg_difflist) == True and abs(C_maxarg_difflist[0]) > 1e-16:
        period=C_maxarg_difflist[0]
        periodic=1
    elif len(C_minarg_difflist) >= 2 and all(j == C_minarg_difflist[0] for j in C_minarg_difflist) == True and abs(C_minarg_difflist[0]) > 1e-16:
        period=C_minarg_difflist[0]
        periodic=1
    else:
        period=len(set(np.round(z_C_Wout_alpha_maxset,rounding_no)))
        periodic=0
    
    return period,periodic,z_C_Wout_alpha_maxset,z_C_Wout_alpha_minset

def test_err_analysis(data, sample_start, sample_end, stepback, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no):
    
    if all(i < FP_err_lim for i in abs(np.diff(data[:,1][-1000::50]))) == True:
        err_C = 3.0#Fixed Point
        Cest_radius=999
        c_roundness=999
        
        C_xcenter_err=999
        C_ycenter_err=999
        x_C_no_of_unique_maxima=999
        C_periodic_prof=999
        xmax_localmaxima_C=999
        xmin_localmaxima_C=999
        xmax_localminima_C=999
        xmin_localminima_C=999
        ymax_localmaxima_C=999
        ymin_localmaxima_C=999
        ymax_localminima_C=999
        ymin_localminima_C=999
    else:
        center, localmax_x_pos, localmax_y_pos, localmin_x_pos, localmin_y_pos = estimate_circle_center(data, sample_start, sample_end)
        rotation_direction = direction_of_rotation_stricter(data, localmax_x_pos, localmax_y_pos, localmin_x_pos, localmin_y_pos, stepback)
        c_roundness = roundness(data, localmax_x_pos, center)

        _, Ckdifflist = list_to_check_if_LC(data[:,1], sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol)

        x_C_no_of_unique_maxima,C_periodic_prof,x_C_localmaxima_v3,x_C_localminima_v3 = list_to_check_if_LC_v3(data[:,0], sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no)
        xmax_localmaxima_C=max(x_C_localmaxima_v3)
        xmin_localmaxima_C=min(x_C_localmaxima_v3)
        xmax_localminima_C=max(x_C_localminima_v3)
        xmin_localminima_C=min(x_C_localminima_v3)
        
        y_C_no_of_unique_maxima,y_C_periodic_prof,y_C_localmaxima_v3,y_C_localminima_v3 = list_to_check_if_LC_v3(data[:,1], sample_start, sample_end, FP_err_lim, FP_sample_start, FP_sample_end, LC_err_tol, rounding_no)
        ymax_localmaxima_C=max(y_C_localmaxima_v3)
        ymin_localmaxima_C=min(y_C_localmaxima_v3)
        ymax_localminima_C=max(y_C_localminima_v3)
        ymin_localminima_C=min(y_C_localminima_v3)

        if len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and rotation_direction==1:
            err_C = 2.0#LC rotating in anti-clockwise direction CA/C1
        elif len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and rotation_direction==-1:
            err_C = 5.0#LC rotating in clockwise direction CB/C2
        elif len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and rotation_direction==0:
            err_C = 6.0#LC changing direction of rotation
        elif rotation_direction==1:
            err_C = 7.0
        elif rotation_direction==-1:
            err_C = 8.0
        elif rotation_direction==0:
            err_C = 9.0
        else:
            err_C = 4.0

    return err_C,c_roundness, \
            xmax_localmaxima_C,xmin_localmaxima_C,xmax_localminima_C,xmin_localminima_C, \
            ymax_localmaxima_C,ymin_localmaxima_C,ymax_localminima_C,ymin_localminima_C

def check_err_maxminCA(err_C,xmax_localmaxima_C,ymax_localmaxima_C,xmax_localminima_C,ymax_localminima_C,xmin_localmaxima_C,ymin_localmaxima_C,xmin_localminima_C,ymin_localminima_C):
    xmax_localmax=4.5
    xmax_localmin=4.5
    xmin_localmax=5.5
    xmin_localmin=5.5
    
    err_copy=err_C
    if err_C==2 and abs(xmax_localmaxima_C) <= xmax_localmax:
        err_copy=22.0
    elif err_C==7 and abs(xmax_localmaxima_C) <= xmax_localmax:
        err_copy=77.0
    elif err_C==2 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=22.0
    elif err_C==7 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=77.0
        
    elif err_C==2 and abs(xmax_localminima_C) <= xmax_localmin:
        err_copy=22.0
    elif err_C==7 and abs(xmax_localminima_C) <= xmax_localmin:
        err_copy=77.0
    elif err_C==2 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=22.0
    elif err_C==7 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=77.0
    
    elif err_C==2 and abs(xmin_localmaxima_C) >= xmin_localmax:
        err_copy=22.0
    elif err_C==7 and abs(xmin_localmaxima_C) >= xmin_localmax:
        err_copy=77.0
    elif err_C==2 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=22.0
    elif err_C==7 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=77.0
    
    elif err_C==2 and abs(xmin_localminima_C) >= xmin_localmin:
        err_copy=22.0
    elif err_C==7 and abs(xmin_localminima_C) >= xmin_localmin:
        err_copy=77.0
    elif err_C==2 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=22.0
    elif err_C==7 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=77.0
    
    else:
        err_copy=4.0
    return err_copy

##Error analysis function for final filter of circle errors
def check_err_maxminCB(err_C,xmax_localmaxima_C,ymax_localmaxima_C,xmax_localminima_C,ymax_localminima_C,xmin_localmaxima_C,ymin_localmaxima_C,xmin_localminima_C,ymin_localminima_C):
    xmax_localmax=4.5
    xmax_localmin=4.5
    xmin_localmax=5.5
    xmin_localmin=5.5
    
    err_copy=err_C
    if err_C==5 and abs(xmax_localmaxima_C) <= xmax_localmax:
        err_copy=55.0
    elif err_C==8 and abs(xmax_localmaxima_C) <= xmax_localmax:
        err_copy=88.0
    elif err_C==5 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=55.0
    elif err_C==8 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=88.0
        
    elif err_C==5 and abs(xmax_localminima_C) <= xmax_localmin:
        err_copy=55.0
    elif err_C==8 and abs(xmax_localminima_C) <= xmax_localmin:
        err_copy=88.0
    elif err_C==5 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=55.0
    elif err_C==8 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=88.0
    
    elif err_C==5 and abs(xmin_localmaxima_C) >= xmin_localmax:
        err_copy=55.0
    elif err_C==8 and abs(xmin_localmaxima_C) >= xmin_localmax:
        err_copy=88.0
    elif err_C==5 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=55.0
    elif err_C==8 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=88.0
    
    elif err_C==5 and abs(xmin_localminima_C) >= xmin_localmin:
        err_copy=55.0
    elif err_C==8 and abs(xmin_localminima_C) >= xmin_localmin:
        err_copy=88.0
    elif err_C==5 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=55.0
    elif err_C==8 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=88.0
    
    else:
        err_copy=4.0
    return err_copy

def get_error_both(c1_rel_roundness, c2_rel_roundness, err_c1, err_c2, filt_err_c1, filt_err_c2, lc_error_bound):
    err_vals_CA = []
    err_vals_CB = []

    for i in range(len(err_c1)):
        dummy = err_CA[i]
        if dummy == 2.0 and CArel_roundness[i] <= LC_error_bound and filt_err_CA[i] == 4.0:
            dummy = CArel_roundness[i]
        else:
            dummy = np.nan
        err_vals_CA.append(dummy)
    
    for i in range(len(err_CB)):
        dummy = err_CB[i]
        if dummy == 5.0 and CBrel_roundness[i] <= LC_error_bound and filt_err_CB[i] == 4.0:
            dummy = CBrel_roundness[i]
        else:
            dummy = np.nan
        err_vals_CB.append(dummy)
    
    err_vals_both=np.array(list(zip(err_vals_CA,err_vals_CB)))
    err_both=np.array(list(zip(err_CA,err_CB)))

    good_pair=np.array([[2,5]])
    maxerr = np.zeros(len(err_both))
    for i in range(len(err_both)):
        for pair in good_pair:
            pair=np.array([pair[0],pair[1]])
            #print(pair,err_both[i])
            if pair[0] == err_both[i][0] and pair[1] == err_both[i][1] and CArel_roundness[i] <= LC_error_bound and CBrel_roundness[i] <= LC_error_bound and filt_err_CA[i] == 4.0 and filt_err_CB[i] == 4.0:
                maxerr[i] = 1.0#err_both[i]
                break
            else:
                maxerr[i] = 0.0
                
    maxerr_vals = np.empty(len(err_vals_both))
    for i in range(len(err_vals_both)):
        maxerr_vals[i] = np.amax(err_vals_both[i])
        
    return err_vals_both,err_both,maxerr,maxerr_vals