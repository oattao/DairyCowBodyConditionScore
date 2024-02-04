def scoring_front_teat_length(length):
    steps = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    scores = [9, 8, 7, 6, 5, 4, 3, 2]
    for (step, sc) in zip(steps, scores):
        if length > step:
            return sc
    return 1

def scoring_rump_angle(delta):
    steps = [12, 10, 8, 6, 4, 2, 0, -2]
    scores = [9, 8, 7, 6, 5, 4, 3, 2]
    for (step, sc) in zip(steps, scores):
        if delta > step:
            return sc
    return 1

def scoring_rump_width(rump_width):
    rump_width *= 100 # cm
    steps = [17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if rump_width > step:
            return sc
    return 1

def scoring_stature(height):
    # height *= 100 # to cm
    steps = [152, 149, 146, 143, 140, 137, 134, 131, 128]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if height >= step:
            return sc
    return 1

def scoring_udder_depth(delta):
    # delta = (z_udder - z_knee) * 100 # cm
    steps = [18, 15, 12, 9, 6, 3, 0, -3, -6]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if delta > step:
            return sc
    return 1

def scoring_rump_width(rump_width):
    # rump_width = abs(point_x[1] - point_y[1])
    # rump_width *= 100 # cm
    steps = [17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if rump_width > step:
            return sc
    return 1



