import cv2


def build_gaussian_pyramid(image, levels=9):
    pyramid = [image]
    current = image.copy()

    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)

    return pyramid
