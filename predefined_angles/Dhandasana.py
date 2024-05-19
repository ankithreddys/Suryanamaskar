def Dhandasana(left_elbow,right_elbow, left_knee,right_knee,left_hip,right_hip):
    threshold = 20
    if(
        ((85-threshold <= left_elbow <= 85+threshold) or (85-threshold <= right_elbow <= 85+threshold)) and
        ((180-threshold <= left_knee <= 180+threshold) or (180-threshold <= right_knee <= 180+threshold)) and 
        ((170-threshold <= left_hip <= 170+threshold) or (170-threshold <= right_hip <= 170+threshold))
    ):
        return 'good'
    else:
        if left_hip < 180-threshold or right_hip < 180-threshold:
            return 'lower your hip'
        if right_hip > 180+threshold or left_hip > 180+threshold:
            return 'rise your hip'
        if left_elbow > 85+threshold or right_elbow > 85+threshold:
            return 'lower your elbows a bit backward'
        else:
            return 'keep your knees straight'