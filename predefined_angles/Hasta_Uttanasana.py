def HastaUttasana(left_hip, right_hip, left_elbow, right_elbow,left_shoulder,right_shoulder):
    threshold = 20
    if( ((200-threshold <= left_hip <= 200+threshold) or (200-threshold <= right_hip <= 200+threshold)) and
       ((160-threshold <= left_elbow <= 160+threshold) or (160-threshold <= right_elbow <= 160+threshold)) and
        ((170-threshold <= left_shoulder <= 170+threshold) or (170-threshold <= right_shoulder <= 170+threshold))
    ):
        return 'good'
    else:
        if left_hip < 200-threshold or right_hip < 200-threshold:
            return 'bend your hip back'
        if left_elbow < 160-threshold or left_elbow > 160+threshold or right_elbow < 160-threshold or right_elbow > 160+threshold:
            return 'straight your elbow'
        else:
            return 'other'