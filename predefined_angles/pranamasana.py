def pranamasana(left_elbow, right_elbow, left_knee, right_knee, left_hip, right_hip):
    threshold = 10
    ''''
    left_elbow and right elbow (although should have same angles) are different as one angle is calculated 
    in anti-clockwise and the other in clockwise
    '''
    if(((320-threshold <= left_elbow <= 320+threshold) or (35-threshold <= right_elbow <= 35+threshold)) and
       ((180-threshold <= left_hip <= 180+threshold) or (180-threshold <= right_hip <= 180+threshold) ) 
       ):
        return 'good'
    else:
        if (left_elbow < 320- threshold) and (right_elbow > 35+threshold):
            return 'lift your praying hands slightly up'
        if (left_elbow > 320- threshold) and (right_elbow < 35+threshold):
            return 'lower your praying hands lower '
        
        return 'other'
