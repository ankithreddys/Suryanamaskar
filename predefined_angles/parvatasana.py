def parvatasana(left_elbow,right_elbow,left_hip, right_hip, left_knee, right_knee):
    threshold = 20
    if (((180-threshold<=left_elbow<= 180+threshold) or (180-threshold <= right_elbow <= 180+threshold)) and
        ((70-threshold<=left_hip<=70+threshold) or (70-threshold<=right_hip<=70+threshold)) and
        ((180-threshold<=left_knee<=180+threshold) or (180-threshold<=right_knee<=180+threshold))
        ):
        return 'good'
    else:
        if (left_elbow < 180-threshold) or (right_elbow < 180-threshold):
            return 'straight your elbow'
        if (left_elbow > 180+threshold) or (right_elbow > 180+threshold):
            return 'straight your elbow'
        if (left_hip < 70-threshold) or (right_hip < 70-threshold ):
            return 'lower your hip '
        if (left_hip > 70+threshold) or (right_hip > 70+threshold):
            return 'lift your hip'
        if (left_knee < 180-threshold) or (right_knee < 180-threshold):
            return 'straight your knee'
        if (left_knee > 180+threshold) or (right_knee > 180+threshold):
            return 'straight your knee'