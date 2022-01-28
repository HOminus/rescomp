import numpy as np
import numpy.linalg as npl
from numpy.linalg import inv
from numpy import random
import scipy as scipy
import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from scipy.sparse import identity
from scipy.sparse import vstack

def max_of_three(triple):
    l,m,r = triple
    if l<m and r<m:
        return True
    return False

def min_of_three(triple):
    l,m,r = triple
    if l>m and r>m:
        return True
    return False

def distance_2pts(X,Y):
    return np.sqrt( (Y[0]-X[0])**2 + (Y[1]-X[1])**2 )

def tri_area(X,Y,Z):
    return 0.5*abs( ((Y[0]-X[0])*(Z[1]-Y[1])) - ((Y[1]-X[1])*(Z[0]-Y[0])) )
    #return 0.5*abs(abs(X[0])*(abs(Y[1]-Z[1])) + abs(Y[0])*(abs(Z[1]-X[1])) + abs(Z[0])*(abs(X[1]-Y[1])))

def curvature(X,Y,Z):
    return ( 4*tri_area(X,Y,Z) )/( (distance_2pts(X,Y))*(distance_2pts(Y,Z))*(distance_2pts(Z,X)) )

def list_to_check_if_LC(x,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol):
    z_C1_Wout_alpha = x[sample_start:sample_end]
    z_C1_Wout_alpha_set = []
    for i in range(1,len(z_C1_Wout_alpha)-1):
        if max_of_three(z_C1_Wout_alpha[i-1:i+2]):
            z_C1_Wout_alpha_set.append(z_C1_Wout_alpha[i+1])
    Cklist = []
    if len(z_C1_Wout_alpha_set) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        zmax1 = np.amax(z_C1_Wout_alpha_set)
        for i in range(len(z_C1_Wout_alpha_set)):
            ztest1 = z_C1_Wout_alpha_set[i]
            if abs((ztest1 - zmax1)/zmax1) <= LC_err_tol:
                Cklist.append(i)
    else:
        Cklist.append(0.0)
        Cklist.append(10.0)
        Cklist.append(100.0)
        Cklist.append(1000.0)
    Ckdifflist=np.diff(Cklist)#
    return Cklist,Ckdifflist

def estimate_circ_center(x1,x2,sample_start,sample_end):
    zx_C1_alpha = x1[sample_start:sample_end]
    zx_C1_maxset_alpha = []
    C1x_alpha_localmax_pos=[]
    zx_C1_minset_alpha = []
    C1x_alpha_localmin_pos=[]
    for i in range(1,len(zx_C1_alpha)-1):
        if max_of_three(zx_C1_alpha[i-1:i+2]):
            zx_C1_maxset_alpha.append(zx_C1_alpha[i+1])
            C1x_alpha_localmax_pos.append(sample_start+i+1)
            #print(15000+i+1,',',zx_C1_alpha[i+1])
        elif min_of_three(zx_C1_alpha[i-1:i+2]):
            zx_C1_minset_alpha.append(zx_C1_alpha[i+1])
            C1x_alpha_localmin_pos.append(sample_start+i+1)
            #print(15000+i+1,',',zx_C1_alpha[i+1])
            
    zy_C1_alpha = x2[sample_start:sample_end]
    zy_C1_maxset_alpha = []
    C1y_alpha_localmax_pos=[]
    zy_C1_minset_alpha = []
    C1y_alpha_localmin_pos=[]
    for i in range(1,len(zy_C1_alpha)-1):
        if max_of_three(zy_C1_alpha[i-1:i+2]):
            zy_C1_maxset_alpha.append(zy_C1_alpha[i+1])
            C1y_alpha_localmax_pos.append(sample_start+i+1)
            #print(15000+i+1,',',zy_C1_alpha[i+1])
        elif min_of_three(zy_C1_alpha[i-1:i+2]):
            zy_C1_minset_alpha.append(zy_C1_alpha[i+1])
            C1y_alpha_localmin_pos.append(sample_start+i+1)
            #print(15000+i+1,',',zy_C1_alpha[i+1])
        
    center_est=np.array([(np.average(zx_C1_maxset_alpha)+np.average(zx_C1_minset_alpha))/2,(np.average(zy_C1_maxset_alpha)+np.average(zy_C1_minset_alpha))/2])
        
    return center_est,C1x_alpha_localmax_pos,C1y_alpha_localmax_pos,C1x_alpha_localmin_pos,C1y_alpha_localmin_pos

def direction_of_rotation(x2,localmax_pos,stepback):
    C_vel=[]
    for i in(localmax_pos):
        C_vel.append((x2[i]-x2[i-stepback])/stepback)
    
    if all(i > 0 for i in C_vel) == True:
        C_vel_dir=1 #anticlockwise
    elif all(i < 0 for i in C_vel) == True:
        C_vel_dir=-1 #clockwise
    else:
        C_vel_dir=0 #n/a
    C_vel=np.array(C_vel)
    return C_vel_dir,C_vel

def direction_of_rotation_stricter(x1,x2,x1localmax_pos,x2localmax_pos,x1localmin_pos,x2localmin_pos,stepback):
    C_vel_x1max=[]
    for i in(x1localmax_pos):
        if x1[i]>0:
            C_vel_x1max.append((x2[i]-x2[i-stepback])/stepback)
    C_vel_x1min=[]
    for i in(x1localmin_pos):
        if x1[i]<0:
            C_vel_x1min.append((x2[i]-x2[i-stepback])/stepback)
    C_vel_x2max=[]
    for i in(x2localmax_pos):
        if x2[i]>0:
            C_vel_x2max.append((x1[i]-x1[i-stepback])/stepback)
    C_vel_x2min=[]
    for i in(x2localmin_pos):
        if x2[i]<0:
            C_vel_x2min.append((x1[i]-x1[i-stepback])/stepback)
    
    if all(i > 0 for i in C_vel_x1max) and all(i < 0 for i in C_vel_x1min) and all(i < 0 for i in C_vel_x2max) and all(i > 0 for i in C_vel_x2min) == True:
        C_vel_dir=1 #clockwise
    elif all(i < 0 for i in C_vel_x1max) and all(i > 0 for i in C_vel_x1min) and all(i > 0 for i in C_vel_x2max) and all(i < 0 for i in C_vel_x2min) == True:
        C_vel_dir=-1 #anticlockwise
    else:
        C_vel_dir=0 #n/a
    C_vel_x1max=np.array(C_vel_x1max)
    C_vel_x1min=np.array(C_vel_x1min)
    C_vel_x2max=np.array(C_vel_x2max)
    C_vel_x2min=np.array(C_vel_x2min)
    return C_vel_dir,C_vel_x1max,C_vel_x2max,C_vel_x1min,C_vel_x2min

def calc_curvature_perr(x1,x2,localmax_pos,dd1,randrange,pts23):
    curCstore=[]

    for i in(localmax_pos):
        pp1rand=random.randint(-randrange,randrange)
        pp2rand=random.randint(-randrange,randrange)
        pp3rand=random.randint(-randrange,randrange)
        pp1=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp2=np.array([x1[i+pts23+pp2rand],x2[i+pts23+pp2rand]])
        pp3=np.array([x1[i-pts23+pp3rand],x2[i-pts23+pp3rand]])
        
        curC=curvature(pp1,pp2,pp3)
        curCstore.append(curC)
        #print(cur1)
        
    AvgCcur=np.average(curCstore)
    Ccur_perr=abs(AvgCcur-1/dd1)/(1/dd1)
    return Ccur_perr,AvgCcur

def calc_mean_curvature_perr(x1,x2,xlocalmax_pos,xlocalmin_pos,ylocalmax_pos,ylocalmin_pos,dd1,randrange,pts1,pts2,pts3,pts4):
    curC_xlocalmax_store=[]
    curC_ylocalmax_store=[]
    curC_xlocalmin_store=[]
    curC_ylocalmin_store=[]

    for i in(xlocalmax_pos):
        pp1rand=random.randint(-randrange,randrange)
        pp2rand=random.randint(-randrange,randrange)
        pp3rand=random.randint(-randrange,randrange)
        
        pp11=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp21=np.array([x1[i+pts1+pp2rand],x2[i+pts1+pp2rand]])
        pp31=np.array([x1[i-pts1+pp3rand],x2[i-pts1+pp3rand]])
        
        pp12=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp22=np.array([x1[i+pts2+pp2rand],x2[i+pts2+pp2rand]])
        pp32=np.array([x1[i-pts2+pp3rand],x2[i-pts2+pp3rand]])
        
        pp13=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp23=np.array([x1[i+pts3+pp2rand],x2[i+pts3+pp2rand]])
        pp33=np.array([x1[i-pts3+pp3rand],x2[i-pts3+pp3rand]])
        
        pp14=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp24=np.array([x1[i+pts4+pp2rand],x2[i+pts4+pp2rand]])
        pp34=np.array([x1[i-pts4+pp3rand],x2[i-pts4+pp3rand]])
        
        curC1=curvature(pp11,pp21,pp31)
        curC2=curvature(pp12,pp22,pp32)
        curC3=curvature(pp13,pp23,pp33)
        curC4=curvature(pp14,pp24,pp34)
        curC_xlocalmax_store.append(curC1)
        curC_xlocalmax_store.append(curC2)
        curC_xlocalmax_store.append(curC3)
        curC_xlocalmax_store.append(curC4)
        
    for i in(ylocalmax_pos):
        pp1rand=random.randint(-randrange,randrange)
        pp2rand=random.randint(-randrange,randrange)
        pp3rand=random.randint(-randrange,randrange)
        
        pp11=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp21=np.array([x1[i+pts1+pp2rand],x2[i+pts1+pp2rand]])
        pp31=np.array([x1[i-pts1+pp3rand],x2[i-pts1+pp3rand]])
        
        pp12=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp22=np.array([x1[i+pts2+pp2rand],x2[i+pts2+pp2rand]])
        pp32=np.array([x1[i-pts2+pp3rand],x2[i-pts2+pp3rand]])
        
        pp13=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp23=np.array([x1[i+pts3+pp2rand],x2[i+pts3+pp2rand]])
        pp33=np.array([x1[i-pts3+pp3rand],x2[i-pts3+pp3rand]])
        
        pp14=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp24=np.array([x1[i+pts4+pp2rand],x2[i+pts4+pp2rand]])
        pp34=np.array([x1[i-pts4+pp3rand],x2[i-pts4+pp3rand]])
        
        curC1=curvature(pp11,pp21,pp31)
        curC2=curvature(pp12,pp22,pp32)
        curC3=curvature(pp13,pp23,pp33)
        curC4=curvature(pp14,pp24,pp34)
        curC_ylocalmax_store.append(curC1)
        curC_ylocalmax_store.append(curC2)
        curC_ylocalmax_store.append(curC3)
        curC_ylocalmax_store.append(curC4)
    
    for i in(xlocalmin_pos):
        pp1rand=random.randint(-randrange,randrange)
        pp2rand=random.randint(-randrange,randrange)
        pp3rand=random.randint(-randrange,randrange)
        
        pp11=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp21=np.array([x1[i+pts1+pp2rand],x2[i+pts1+pp2rand]])
        pp31=np.array([x1[i-pts1+pp3rand],x2[i-pts1+pp3rand]])
        
        pp12=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp22=np.array([x1[i+pts2+pp2rand],x2[i+pts2+pp2rand]])
        pp32=np.array([x1[i-pts2+pp3rand],x2[i-pts2+pp3rand]])
        
        pp13=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp23=np.array([x1[i+pts3+pp2rand],x2[i+pts3+pp2rand]])
        pp33=np.array([x1[i-pts3+pp3rand],x2[i-pts3+pp3rand]])
        
        pp14=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp24=np.array([x1[i+pts4+pp2rand],x2[i+pts4+pp2rand]])
        pp34=np.array([x1[i-pts4+pp3rand],x2[i-pts4+pp3rand]])
        
        curC1=curvature(pp11,pp21,pp31)
        curC2=curvature(pp12,pp22,pp32)
        curC3=curvature(pp13,pp23,pp33)
        curC4=curvature(pp14,pp24,pp34)
        curC_xlocalmin_store.append(curC1)
        curC_xlocalmin_store.append(curC2)
        curC_xlocalmin_store.append(curC3)
        curC_xlocalmin_store.append(curC4)
        
    for i in(ylocalmin_pos):
        pp1rand=random.randint(-randrange,randrange)
        pp2rand=random.randint(-randrange,randrange)
        pp3rand=random.randint(-randrange,randrange)
        
        pp11=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp21=np.array([x1[i+pts1+pp2rand],x2[i+pts1+pp2rand]])
        pp31=np.array([x1[i-pts1+pp3rand],x2[i-pts1+pp3rand]])
        
        pp12=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp22=np.array([x1[i+pts2+pp2rand],x2[i+pts2+pp2rand]])
        pp32=np.array([x1[i-pts2+pp3rand],x2[i-pts2+pp3rand]])
        
        pp13=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp23=np.array([x1[i+pts3+pp2rand],x2[i+pts3+pp2rand]])
        pp33=np.array([x1[i-pts3+pp3rand],x2[i-pts3+pp3rand]])
        
        pp14=np.array([x1[i+pp1rand],x2[i+pp1rand]])
        pp24=np.array([x1[i+pts4+pp2rand],x2[i+pts4+pp2rand]])
        pp34=np.array([x1[i-pts4+pp3rand],x2[i-pts4+pp3rand]])
        
        curC1=curvature(pp11,pp21,pp31)
        curC2=curvature(pp12,pp22,pp32)
        curC3=curvature(pp13,pp23,pp33)
        curC4=curvature(pp14,pp24,pp34)
        curC_ylocalmin_store.append(curC1)
        curC_ylocalmin_store.append(curC2)
        curC_ylocalmin_store.append(curC3)
        curC_ylocalmin_store.append(curC4)
        
    AvgCcur_xmax=np.average(curC_xlocalmax_store)
    AvgCcur_ymax=np.average(curC_ylocalmax_store)
    AvgCcur_xmin=np.average(curC_xlocalmin_store)
    AvgCcur_ymin=np.average(curC_ylocalmin_store)
    
    AvgCcur = (AvgCcur_xmax+AvgCcur_ymax+AvgCcur_xmin+AvgCcur_ymin)/4
    
    Ccur_xmax_perr=abs(AvgCcur_xmax-1/dd1)/(1/dd1)
    Ccur_ymax_perr=abs(AvgCcur_ymax-1/dd1)/(1/dd1)
    Ccur_xmin_perr=abs(AvgCcur_xmin-1/dd1)/(1/dd1)
    Ccur_ymin_perr=abs(AvgCcur_ymin-1/dd1)/(1/dd1)
    
    Ccur_perr = (Ccur_xmax_perr+Ccur_ymax_perr+Ccur_xmin_perr+Ccur_ymin_perr)/4
    
    return Ccur_perr,AvgCcur

def calc_orbital_radius_perr(x1,x2,true_radius,true_xcentre,true_ycentre,sample_start,sample_end,iter_no):
    Rad_list=[]
    for i in range(0,iter_no):
        rp1=random.randint(sample_start,sample_end)
        Rad_est=distance_2pts(np.array([true_xcentre,true_ycentre]),np.array([x1[rp1],x2[rp1]]))
        Rad_list.append(Rad_est)
    Rad_avg=np.average(Rad_list)
    Rad_perr = abs(Rad_avg-true_radius)/true_radius
    return Rad_avg,Rad_perr

def roundness(x,y,localmax_pos,Xcen,Ycen):
    res = np.array(list(zip(localmax_pos, localmax_pos[1:])))
    test_dist_avg=[]
    for i in range(len(res)):
        test_dist=distance_2pts([x[res[i][0]:res[i][1]],y[res[i][0]:res[i][1]]],[Xcen,Ycen])
        roundness=np.amax(test_dist)-np.amin(test_dist)
        test_dist_avg.append(roundness)
        #print(np.amax(test_dist),np.amin(test_dist),roundness)
    return np.average(test_dist_avg)

def testing_list_to_check_if_LC(x,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no):
    z_C1_Wout_alpha = x[sample_start:sample_end]#+10*np.ones((sample_end-sample_start))
    z_C1_Wout_alpha_set = []
    z_set_args=[]
    for i in range(1,len(z_C1_Wout_alpha)-1):
        if max_of_three(z_C1_Wout_alpha[i-1:i+2]):
            z_C1_Wout_alpha_set.append(z_C1_Wout_alpha[i+1])
            z_set_args.append(sample_start+i)
    Cklist = []
    if len(z_C1_Wout_alpha_set) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        a_set=set(np.round(z_C1_Wout_alpha_set,rounding_no))
        Cklist.append(len(a_set))
    else:
        Cklist.append(404.0)
        #Cklist.append(10.0)
        #Cklist.append(100.0)
        #Cklist.append(1000.0)
    #Ckdifflist=np.diff(Cklist)#
    return Cklist,z_C1_Wout_alpha_set

def list_to_check_if_LC_v3(x,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no):
    z_C1_Wout_alpha = x[sample_start:sample_end]
    z_C1_Wout_alpha_maxset = []
    z_C1_Wout_alpha_minset = []
    for i in range(1,len(z_C1_Wout_alpha)-1):
        if max_of_three(z_C1_Wout_alpha[i-1:i+2]):
            z_C1_Wout_alpha_maxset.append(z_C1_Wout_alpha[i+1])
        elif min_of_three(z_C1_Wout_alpha[i-1:i+2]):
            z_C1_Wout_alpha_minset.append(z_C1_Wout_alpha[i+1])
            
    C_maxarg_list = []
    C_minarg_list = []
    if len(z_C1_Wout_alpha_maxset) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        zmax1 = np.amax(z_C1_Wout_alpha_maxset)
        for i in range(len(z_C1_Wout_alpha_maxset)):
            ztest1 = z_C1_Wout_alpha_maxset[i]
            if abs((ztest1 - zmax1)/zmax1) <= LC_err_tol:
                C_maxarg_list.append(i)
    else:
        C_maxarg_list.append(0.0)
        C_maxarg_list.append(10.0)
        C_maxarg_list.append(100.0)
        C_maxarg_list.append(1000.0)
    C_maxarg_difflist=np.diff(C_maxarg_list)
    
    if len(z_C1_Wout_alpha_minset) != 0 and all(i < FP_err_lim for i in abs(np.diff(x[FP_sample_start:FP_sample_end:50])))==False:
        zmin1 = np.amin(z_C1_Wout_alpha_minset)
        for i in range(len(z_C1_Wout_alpha_minset)):
            ztest1 = z_C1_Wout_alpha_minset[i]
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
        period=len(set(np.round(z_C1_Wout_alpha_maxset,rounding_no)))
        periodic=0
    
    return period,periodic,z_C1_Wout_alpha_maxset,z_C1_Wout_alpha_minset

def period_from_maxima_minima_v3(x_max,x_min,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no):
    C_maxarg_list = []
    C_minarg_list = []
    if len(x_max) != 0:
        zmax1 = np.amax(x_max)
        for i in range(len(x_max)):
            ztest1 = x_max[i]
            if abs((ztest1 - zmax1)/zmax1) <= LC_err_tol:
                C_maxarg_list.append(i)
    else:
        C_maxarg_list.append(0.0)
        C_maxarg_list.append(10.0)
        C_maxarg_list.append(100.0)
        C_maxarg_list.append(1000.0)
    C_maxarg_difflist=np.diff(C_maxarg_list)
    
    if len(x_min) != 0:
        zmax1 = np.amax(x_min)
        for i in range(len(x_min)):
            ztest1 = x_min[i]
            if abs((ztest1 - zmin1)/zmin1) <= LC_err_tol:
                C_maxarg_list.append(i)
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
        period=len(set(np.round(x_max,rounding_no)))
        periodic=0
    
    return period,periodic

##Error analysis function for prediction of a given cycle
def Error_analysis_of_Pred_Circle(xpredict1_MF,xpredict2_MF,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,sample_start,sample_end,stepback,dd1,randrange,pts1,pts2,pts3,pts4,xcen,ycen,iter_no):
    if all(i < FP_err_lim for i in abs(np.diff(xpredict2_MF[19000:20000:50]))) == True:
        err_C1 = 3.0
        C1est_radius=999
        C1_roundness=999
        C1cur_perr=999
        C1est_mean_radius=999
        C1cur_mean_perr=999
        C1_Rad_avg=999
        C1_Rad_perr=999
        C1_xcenter_err=999
        C1_ycenter_err=999
    else:
        C1_center_est,C1x_alpha_localmax_pos,C1y_alpha_localmax_pos,C1x_alpha_localmin_pos,C1y_alpha_localmin_pos=estimate_circ_center(xpredict1_MF,xpredict2_MF,sample_start,sample_end)
        C1_xcenter_err=abs(Xcen1-C1_center_est[0])#/C1_center_true[0]
        C1_ycenter_err=abs(0-C1_center_est[1])#/C1_center_true[1]
        C1_vel_dir,C1_vel=direction_of_rotation(xpredict2_C1_MF,C1x_alpha_localmax_pos,stepback)
        C1_vel_dir_strict,C1_vel_x1max_strict,C1_vel_x2max_strict,C1_vel_x1min_strict,C1_vel_x2min_strict=direction_of_rotation_stricter(xpredict1_MF,xpredict2_MF,C1x_alpha_localmax_pos,C1y_alpha_localmax_pos,C1x_alpha_localmin_pos,C1y_alpha_localmin_pos,stepback)
        C1cur_perr,C1Avg_cur=calc_curvature_perr(xpredict1_MF,xpredict2_MF,C1x_alpha_localmax_pos,dd1,randrange,pts1)
        C1est_radius=1/C1Avg_cur
        C1cur_mean_perr,MeanC1cur=calc_mean_curvature_perr(xpredict1_MF,xpredict2_MF,C1x_alpha_localmax_pos,C1y_alpha_localmax_pos,C1x_alpha_localmin_pos,C1y_alpha_localmin_pos,dd1,randrange,pts1,pts2,pts3,pts4)
        C1est_mean_radius=1/MeanC1cur
        C1_Rad_avg,C1_Rad_perr=calc_orbital_radius_perr(xpredict1_MF,xpredict2_MF,dd1,Xcen1,ycen,sample_start,sample_end,iter_no)
        C1_roundness=roundness(xpredict1_MF,xpredict2_MF,C1x_alpha_localmax_pos,Xcen1,ycen)
        C1klist,C1kdifflist=list_to_check_if_LC(xpredict2_MF,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol)
        
        if len(C1kdifflist) >= 2 and all(j == C1kdifflist[0] for j in C1kdifflist) == True and abs(C1kdifflist[0]) > 1e-16 and C1_vel_dir_strict==1:
            err_C1 = 2.0
        elif len(C1kdifflist) >= 2 and all(j == C1kdifflist[0] for j in C1kdifflist) == True and abs(C1kdifflist[0]) > 1e-16 and C1_vel_dir_strict==-1:
            err_C1 = 5.0
        elif len(C1kdifflist) >= 2 and all(j == C1kdifflist[0] for j in C1kdifflist) == True and abs(C1kdifflist[0]) > 1e-16 and C1_vel_dir_strict==0:
            err_C1 = 6.0
        elif C1_vel_dir_strict==1:
            err_C1 = 7.0
        elif C1_vel_dir_strict==-1:
            err_C1 = 8.0
        elif C1_vel_dir_strict==0:
            err_C1 = 9.0
        else:
            err_C1 = 4.0
    
    
    return err_C1,C1est_radius,C1_roundness,C1cur_perr,C1est_mean_radius,C1cur_mean_perr,C1_Rad_avg,C1_Rad_perr,C1_xcenter_err,C1_ycenter_err

##Error analysis function for prediction of a given cycle
def test_Error_analysis_of_Pred_Circle(xpredict1_MF,xpredict2_MF,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no,sample_start,sample_end,stepback,dd1,xcen,ycen,iter_no):
    '''
    
    Function to assess whether the MF RC's estimate of both circles
    were in fact on a circle which rotated in the correct direction and provide an estimate of the roundness
    
    xpredict1_MF: x,y trajectory on circle 1
    xpredict2_MF: x,y trajectory on circle 2
    FP_err_lim: Error threshold on whether a given trajectory is on a FP
    FP_sample_start: Starting time of FP check
    FP_sample_end: Finish time of FP check
    LC_err_tol: Used in the algorithm which checks if a given trajectroy is periodic
    rounding_no: Rounding list vals
    sample_start: Starting time of LC check
    sample_end: Finish time of LC check
    stepback: Used in assessing the direction of rotation
    dd1: radius of the true circle
    xcen: x center of true circle
    ycen: y center of true circle
    iter_no: used in calculating the percentage error in actual v predicted radius
    
    
    '''
    #This if statement assigns dummy variables to a trajectory on a fixed point
    #The else statement below caters for trajectories on limit cycles, chaotic attractors, etc.
    if all(i < FP_err_lim for i in abs(np.diff(xpredict2_MF[-1000::50]))) == True:
        err_C = 3.0#Fixed Point
        Cest_radius=999
        C_roundness=999
        C_vel_dir_strict=999
        C_Rad_perr=999
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
        C_center_est,Cx_alpha_localmax_pos,Cy_alpha_localmax_pos,Cx_alpha_localmin_pos,Cy_alpha_localmin_pos=estimate_circ_center(xpredict1_MF,xpredict2_MF,sample_start,sample_end)
        C_xcenter_err=abs(xcen-C_center_est[0])#/C_center_true[0]
        C_ycenter_err=abs(ycen-C_center_est[1])#/C_center_true[1]
        #C_vel_dir,C_vel=direction_of_rotation(xpredict2_MF,Cx_alpha_localmax_pos,stepback)
        C_vel_dir_strict,C_vel_x1max_strict,C_vel_x2max_strict,C_vel_x1min_strict,C_vel_x2min_strict=direction_of_rotation_stricter(xpredict1_MF,xpredict2_MF,Cx_alpha_localmax_pos,Cy_alpha_localmax_pos,Cx_alpha_localmin_pos,Cy_alpha_localmin_pos,stepback)

        C_Rad_avg,C_Rad_perr=calc_orbital_radius_perr(xpredict1_MF,xpredict2_MF,dd1,xcen,ycen,sample_start,sample_end,iter_no)
        C_roundness=roundness(xpredict1_MF,xpredict2_MF,Cx_alpha_localmax_pos,xcen,ycen)
        Cklist,Ckdifflist=list_to_check_if_LC(xpredict2_MF,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol)
        
        x_C_no_of_unique_maxima,C_periodic_prof,x_C_localmaxima_v3,x_C_localminima_v3=list_to_check_if_LC_v3(xpredict1_MF,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no)
        xmax_localmaxima_C=max(x_C_localmaxima_v3)
        xmin_localmaxima_C=min(x_C_localmaxima_v3)
        xmax_localminima_C=max(x_C_localminima_v3)
        xmin_localminima_C=min(x_C_localminima_v3)
        y_C_no_of_unique_maxima,y_C_periodic_prof,y_C_localmaxima_v3,y_C_localminima_v3=list_to_check_if_LC_v3(xpredict2_MF,sample_start,sample_end,FP_err_lim,FP_sample_start,FP_sample_end,LC_err_tol,rounding_no)
        ymax_localmaxima_C=max(y_C_localmaxima_v3)
        ymin_localmaxima_C=min(y_C_localmaxima_v3)
        ymax_localminima_C=max(y_C_localminima_v3)
        ymin_localminima_C=min(y_C_localminima_v3)
        
        if len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and C_vel_dir_strict==1:
            err_C = 2.0#LC rotating in anti-clockwise direction CA/C1
        elif len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and C_vel_dir_strict==-1:
            err_C = 5.0#LC rotating in clockwise direction CB/C2
        elif len(Ckdifflist) >= 2 and all(j == Ckdifflist[0] for j in Ckdifflist) == True and abs(Ckdifflist[0]) > 1e-16 and C_vel_dir_strict==0:
            err_C = 6.0#LC changing direction of rotation
        elif C_vel_dir_strict==1:
            err_C = 7.0
        elif C_vel_dir_strict==-1:
            err_C = 8.0
        elif C_vel_dir_strict==0:
            err_C = 9.0
        else:
            err_C = 4.0
    
    
    return err_C,C_vel_dir_strict,C_roundness,C_Rad_perr,C_xcenter_err,C_ycenter_err,x_C_no_of_unique_maxima,C_periodic_prof,xmax_localmaxima_C,xmin_localmaxima_C,xmax_localminima_C,xmin_localminima_C,ymax_localmaxima_C,ymin_localmaxima_C,ymax_localminima_C,ymin_localminima_C

##Error analysis function for final filter of circle errors
def check_errmaxminCA(err_C,Xcen,xmax_localmaxima_C,ymax_localmaxima_C,xmax_localminima_C,ymax_localminima_C,xmin_localmaxima_C,ymin_localmaxima_C,xmin_localminima_C,ymin_localminima_C):
    xmax_localmax=4.5
    xmax_localmin=4.5
    xmin_localmax=5.5
    xmin_localmin=5.5
    
    err_copy=err_C
    if err_C==2 and abs(xmax_localmaxima_C-Xcen) <= xmax_localmax:
        err_copy=22.0
    elif err_C==7 and abs(xmax_localmaxima_C-Xcen) <= xmax_localmax:
        err_copy=77.0
    elif err_C==2 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=22.0
    elif err_C==7 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=77.0
        
    elif err_C==2 and abs(xmax_localminima_C-Xcen) <= xmax_localmin:
        err_copy=22.0
    elif err_C==7 and abs(xmax_localminima_C-Xcen) <= xmax_localmin:
        err_copy=77.0
    elif err_C==2 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=22.0
    elif err_C==7 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=77.0
    
    elif err_C==2 and abs(xmin_localmaxima_C-Xcen) >= xmin_localmax:
        err_copy=22.0
    elif err_C==7 and abs(xmin_localmaxima_C-Xcen) >= xmin_localmax:
        err_copy=77.0
    elif err_C==2 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=22.0
    elif err_C==7 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=77.0
    
    elif err_C==2 and abs(xmin_localminima_C-Xcen) >= xmin_localmin:
        err_copy=22.0
    elif err_C==7 and abs(xmin_localminima_C-Xcen) >= xmin_localmin:
        err_copy=77.0
    elif err_C==2 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=22.0
    elif err_C==7 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=77.0
    
    else:
        err_copy=4.0
    return err_copy

##Error analysis function for final filter of circle errors
def check_errmaxminCB(err_C,Xcen,xmax_localmaxima_C,ymax_localmaxima_C,xmax_localminima_C,ymax_localminima_C,xmin_localmaxima_C,ymin_localmaxima_C,xmin_localminima_C,ymin_localminima_C):
    xmax_localmax=4.5
    xmax_localmin=4.5
    xmin_localmax=5.5
    xmin_localmin=5.5
    
    err_copy=err_C
    if err_C==5 and abs(xmax_localmaxima_C-Xcen) <= xmax_localmax:
        err_copy=55.0
    elif err_C==8 and abs(xmax_localmaxima_C-Xcen) <= xmax_localmax:
        err_copy=88.0
    elif err_C==5 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=55.0
    elif err_C==8 and abs(ymax_localmaxima_C) <= xmax_localmax:
        err_copy=88.0
        
    elif err_C==5 and abs(xmax_localminima_C-Xcen) <= xmax_localmin:
        err_copy=55.0
    elif err_C==8 and abs(xmax_localminima_C-Xcen) <= xmax_localmin:
        err_copy=88.0
    elif err_C==5 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=55.0
    elif err_C==8 and abs(ymax_localminima_C) <= xmax_localmin:
        err_copy=88.0
    
    elif err_C==5 and abs(xmin_localmaxima_C-Xcen) >= xmin_localmax:
        err_copy=55.0
    elif err_C==8 and abs(xmin_localmaxima_C-Xcen) >= xmin_localmax:
        err_copy=88.0
    elif err_C==5 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=55.0
    elif err_C==8 and abs(ymin_localmaxima_C) >= xmin_localmax:
        err_copy=88.0
    
    elif err_C==5 and abs(xmin_localminima_C-Xcen) >= xmin_localmin:
        err_copy=55.0
    elif err_C==8 and abs(xmin_localminima_C-Xcen) >= xmin_localmin:
        err_copy=88.0
    elif err_C==5 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=55.0
    elif err_C==8 and abs(ymin_localminima_C) >= xmin_localmin:
        err_copy=88.0
    
    else:
        err_copy=4.0
    return err_copy

def fix_length_of_maxmins_with_nans(rho_steps,R_C_local_max_or_min_data):
    row_lengths_maxC1=[]
    for i in range(rho_steps):
        for row_maxC1 in R_C_local_max_or_min_data[i][1]:
            row_lengths_maxC1.append(len(row_maxC1))

    max_length_maxC1 = max(row_lengths_maxC1)
    print('max length:',max_length_maxC1)
    
    for i in range(rho_steps):
        for row_maxC1 in R_C_local_max_or_min_data[i][1]:
            while len(row_maxC1) < max_length_maxC1:
                row_maxC1.append(None)
    
    return R_C_local_max_or_min_data,max_length_maxC1

def GetErrorBoth(CArel_roundness,CBrel_roundness,err_CA,err_CB,filt_err_CA,filt_err_CB,LC_error_bound):
    err_vals_CA = []
    err_vals_CB = []


    for i in range(len(err_CA)):
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
