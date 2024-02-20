from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, sobel
import scipy
from collections import defaultdict

### Load Image ###
original_image = Image.open("res/WFBGA3.8X4.1-4020160702211656.02.tif")
original = np.asarray(original_image)
W = original.shape[1]
H = original.shape[0]

### Crop the border ###
# original = original[2*H//3:,2*W//3:]
# original = original[:-H//3,:-W//3]
original = original[H//3:,W//3:]

### Select dark pixels (borders) ###
filter_mag = np.percentile(original.reshape(-1), 5)
cond = np.vectorize(lambda x: 1 if x <= filter_mag else 0)
top_masked = cond(original)


### Fourie analysis ###
trans = np.fft.fft2(top_masked)
fft_img = np.fft.fftshift(trans)
req_freq = fft_img.copy()
H = len(req_freq)
W = len(req_freq[0])
for i in range(H):
    for j in range(W):
        if not (np.abs(i - H//2) < 300 and np.abs(j - W//2) < 300):
            req_freq[i][j] = 0
req_freq = req_freq*(np.log(np.abs(fft_img)) > 8)

freq_filt_img = np.fft.ifft2(np.fft.ifftshift(req_freq))
freq_filt_img = np.abs(freq_filt_img)
freq_filt_img = freq_filt_img.astype(np.float32)


### Detect intersections ###
l1 = 100000
l2 = 100000
sigma = 14

while l1 == 100000:
    sigma -= 1
    while np.count_nonzero(gaussian_filter(freq_filt_img, sigma=sigma) > 1) < 4:
        sigma -= 1
    
    print(f"final sigma: {sigma}")
    freq_filt_img_smo = gaussian_filter(freq_filt_img, sigma=sigma)

    positions = defaultdict(list)
    points = freq_filt_img_smo > 1
    result = []
    def dist(x, y):
        return np.sqrt(np.sum(np.square(x-y)))
    dots = np.argwhere(points)
    for dot in dots:
        found = False
        for k, v in positions.items():
            if dist(np.array(k), dot) < 50:
                positions[k].append(dot)
                found = True
        
        if not found:
            positions[(dot[0], dot[1])].append(dot)
    for xpos, yposes in positions.items():
        yposes = np.array(yposes)
        result.append((np.mean(yposes[:,0]), np.mean(yposes[:,1])))
    result = [[int(x[0]), int(x[1])] for x in result]

    ### Get shape parameters ###
    # distances = []
    for d1 in result:
        cxdistances = []
        cydistances = []
        for d2 in result:
            if np.abs(d1[1]-d2[1]) < 30:
                cxdistances.append(np.abs(d1[0]-d2[0]))
            if np.abs(d1[0]-d2[0]) < 30:
                cydistances.append(np.abs(d1[1]-d2[1]))
        cxdistances.sort()
        cydistances.sort()
        cxdistances = cxdistances[1:]
        cydistances = cydistances[1:]
        if len(cxdistances) > 0 and cxdistances[0] < l1:
            l1 = cxdistances[0]

        if len(cydistances) > 0 and cydistances[0] < l2:
            l2 = cydistances[0]

    print(l1, l2)