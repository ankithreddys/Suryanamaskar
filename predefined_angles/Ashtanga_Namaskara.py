def AshtangaNamaskara(left_elbow, right_elbow, left_hip, right_hip, left_knee, right_knee):
    threshold = 30
    if( ((40-threshold <= left_elbow <= 40+threshold) or (40-threshold <= right_elbow <= 40+threshold)) and
       ((100-threshold <= left_hip <= 100+threshold) or (100-threshold <= right_hip <= 100+threshold))

    ):
        return 'good'
    
    else:
        return 'not good'