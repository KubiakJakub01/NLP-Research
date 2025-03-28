def calculate_brightness(img):
    if len(img) == 0 or len(img[0]) == 0:
        return -1
    n = len(img[0])
    avg = 0
    counter = 0
    for row in img:
        if len(row) != n:
            return -1
        for i in row:
            if i < 0 or i > 255:
                return -1
            avg += i
            counter += 1
    return avg / counter
