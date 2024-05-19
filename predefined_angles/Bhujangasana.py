def Bhujangasana(left_knee, right_knee, left_hip, right_hip, left_elbow,right_elbow):
    threshold = 10
    if( ((220-threshold <= left_hip <= 220+threshold) or (220-threshold <= right_hip <= 220+threshold)) and
       ((180-threshold <= left_elbow <= 180+threshold ) or (180-threshold <= right_elbow <= 180+threshold )) and 
       ((215-threshold <= left_knee <= 215+threshold) or (215-threshold <= right_knee <= 215+threshold)) 
    ):
        return 'good'
    else:
        if left_hip < 220-threshold:
            return 'lower your hip'
        if left_elbow < 180-threshold or right_elbow < 180-threshold:
            return 'rise your elbows'
        
        else:
            return 'problem with knees'