
import pkg_resources
pkg_resources.working_set.by_key['pytesseract'].version

"""## Importing Libraries"""

import re
import cv2
import pytesseract
import numpy as np


"""## Reading the Query Image"""

Query = cv2.imread("https://www.google.com/url?sa=i&url=https%3A%2F%2Fcreate.microsoft.com%2Fid-id%2Ftemplates%2Ffaktur&psig=AOvVaw3AVvbzRXjOUf3NXDGWAYZA&ust=1749740631079000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCMjp_vjR6Y0DFQAAAAAdAAAAABAS")

# # Viewing the Query Image
# cv2_imshow(Query)

"""## Initializing ORB keypoint detector"""

orb = cv2.ORB_create(5000) # Increase this number if all features are not being detected
keypoints, descriptors = orb.detectAndCompute(Query, None)

"""## Initializing features to be detected"""

# Instead of manually finding out the bounding box co-ordinates of every single feature,
# I used a script that automates the process. Link attached for reference
# https://www.murtazahassan.com/courses/opencv-projects/lesson/complete-code-files/

Features = [[(369, 123), (875, 209), 'Customer Name'],
            [(369, 212), (879, 256), 'Address'],
            [(369, 318), (874, 363), 'Phone Number'],
            [(1292, 208), (1508, 260), 'Date'],
            [(74, 538), (864, 604), 'Description'],
            [(1042, 555), (1296, 1198), 'Unit Price'],
            [(1298, 540), (1508, 602), 'Amount'],
            [(1301, 1201), (1515, 1246), 'Sub Total'],
            [(1303, 1253), (1516, 1301), 'Tax Rate'],
            [(1304, 1304), (1519, 1351), 'Tax'],
            [(1304, 1354), (1520, 1400), 'Total Amount']]

"""## Defining a function that resizes the input image using cv2.warpPerspective() and crops areas of interests

This function first displays the input image as it is and how it looks after calling cv2.WarpPerspective()
"""

def resize():
    imagepath = "/content/"
    picture = input("Enter name of the image: ")
    image = cv2.imread(imagepath + f"/{picture}")
    w, h, _ = image.shape

    dem = image.copy()
    dem = cv2.resize(dem,(650, 1000))
    print("Input Image: \n")
    cv2_imshow(dem)
    print()

    keypoints_2, descriptors_2 = orb.detectAndCompute(image, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(descriptors_2, descriptors)
    matches = sorted(matches, key = lambda x:x.distance)

    best_matches = matches[:int(len(matches) * 0.25)] # 25% of the best matches

    # # Visualizing the matches
    # pic_match = cv2.drawMatches(Query, keypoints, image, keypoints_2, best_matches[:30], None, flags = 2)
    # pic_match = cv2.resize(pic_match, (w // 2, h // 2))
    # cv2_imshow(pic_match)

    source_points = np.float32([keypoints_2[i.queryIdx].pt for i in best_matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints[i.trainIdx].pt for i in best_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    scanned_image = cv2.warpPerspective(image, homography, (Query.shape[1], Query.shape[0]))

    scanned_copy = scanned_image.copy()
    scanned_mask = np.zeros_like(scanned_copy)
    demo = scanned_image.copy()
    demo = cv2.resize(demo,(Query.shape[1] // 2, Query.shape[0] // 2))
    print("Input Image after being transformed: \n")
    cv2_imshow(demo)
    return scanned_copy, scanned_mask, scanned_image

copy, mask, ogimage = resize()

"""## Looping over features to crop and feed into PyTesseract for OCR detection and printing the results"""

print("PyTesseract detected the following features:")
print()

myfeatures = []

for i, feature in enumerate(Features):

    # # Drawing detected features
    # cv2.rectangle(mask, (feature[0][0], feature[0][1]), (feature[1][0], feature[1][1]), (0, 255, 255), cv2.FILLED)
    # copy = cv2.addWeighted(copy, 0.99, mask, 0.1, 0)

    crop = ogimage[feature[0][1] : feature[1][1], feature[0][0] : feature[1][0]] # Extracting features to feed into pytesseract

    # Viewing the crops
    # cv2_imshow(crop)

    print("{}: {}".format(str(feature[2]), re.sub(r'[^A-Za-z0-9/%]+', ' ', pytesseract.image_to_string(crop))))

    myfeatures.append(pytesseract.image_to_string(crop))
    myfeatures_clean = [re.sub(r'[^A-Za-z0-9,/%]+', ' ', x) for x in myfeatures]
    cv2.putText(copy, str(myfeatures_clean[i]), (feature[0][0], feature[0][1] - 20),
                    cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 50, 255), 3)

"""## Displaying the same image with text overlaying the detected features"""

w, h, _ = copy.shape
copy = cv2.resize(copy, (w // 2, h // 2))
cv2_imshow(copy)