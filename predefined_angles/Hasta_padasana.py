def HastaPadasana(left_hip, right_hip, left_elbow, right_elbow, left_knee, right_knee):
    threshold = 15
    if (
        ((45-threshold <= left_hip <= 45+threshold) or (45-threshold <= right_hip <= 45+threshold)) and
        ((180-threshold <= left_elbow <= 180+threshold) or (180-threshold <= right_elbow <= 180+threshold)) and
        ((190-threshold <= left_knee <= 190+threshold) or (190-threshold <= right_knee <= 190+threshold))
    ):
        return 'good'
    
    else:
        if left_hip > 45+threshold or right_hip > 45+threshold:
            return 'lower your hip'
        if left_hip < 45-threshold or right_hip < 45-threshold:
            return 'rise your hip'
        if left_knee > 190+threshold or right_knee > 190+threshold:
            return 'straight your knees by bending them slightly back'
        if left_knee < 190-threshold or right_knee < 190-threshold:
            return 'staight your knees by moving them slighty forward'
        if left_elbow < 180-threshold or right_elbow < 180-threshold:
            return 'straight your elbows by moving them slightly back'
        if left_elbow > 180+threshold or right_elbow > 180+ threshold:
            return 'straight your elbows by moving them slightly front'
