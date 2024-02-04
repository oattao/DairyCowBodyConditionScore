from .cloud import compute_angle


def scoring_rear_leg_rear_view(rear_angle):
    steps = [15, 1, 0]
    scores = [9, 5, 1]
    for (step, sc) in zip(steps, scores):
        if rear_angle > step:
            return sc, rear_angle
    return 9, rear_angle

def scoring_rear_leg_side_view(side_angle):
    steps = [20, 15, 5, 1, 0]
    scores = [9, 7, 5, 3, 1]
    for (step, sc) in zip(steps, scores):
        if side_angle > step:
            return sc, side_angle
    return 1, side_angle

def scoring_front_teat_length(length1, length2):
    length = (length1 + length2)*0.5
    length *= 100 # to cm
    steps = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    scores = [9, 8, 7, 6, 5, 4, 3, 2]
    for (step, sc) in zip(steps, scores):
        if length > step:
            return sc, length
    return 1, length

def scoring_front_teat_replacement(line_params1, line_params2):
    normal1 = line_params1[1]
    normal2 = line_params2[1]
    angle = compute_angle(normal1, normal2)
    if angle < 5:
        return 5, angle
    if angle < 10:
        return 6, angle
    elif angle < 20:
        return 3, angle
    return 5, angle

def scoring_rump_angle(point_a, point_b):
    delta = point_a[2] - point_b[2]
    delta *= 100
    steps = [12, 10, 8, 6, 4, 2, 0, -2]
    scores = [9, 8, 7, 6, 5, 4, 3, 2]
    for (step, sc) in zip(steps, scores):
        if delta > step:
            return sc, delta
    return 1, delta

def scoring_rump_width(rump_width):
    rump_width *= 100 # cm
    steps = [17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if rump_width > step:
            return sc, rump_width
    return 1, rump_width

def scoring_stature(height):
    # height *= 100 # to cm
    steps = [152, 149, 146, 143, 140, 137, 134, 131, 128]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if height >= step:
            return sc
    return 1

def scoring_udder_depth(z_knee, z_udder):
    delta = (z_udder - z_knee) * 100 # cm
    steps = [18, 15, 12, 9, 6, 3, 0, -3, -6]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if delta > step:
            return sc, delta
    return 1, delta

def scoring_rump_width(point_x, point_y):
    rump_width = abs(point_x[1] - point_y[1])
    rump_width *= 100 # cm
    steps = [17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5]
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    for (step, sc) in zip(steps, scores):
        if rump_width > step:
            return sc, rump_width
    return 1, rump_width

