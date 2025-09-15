import importlib
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters, measure, segmentation, exposure
from skimage import morphology
from skimage.color import rgb2gray
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage import filters, morphology, measure
import cv2
from sklearn.cluster import HDBSCAN
import pandas as pd
import math



def load_and_preprocess(image_path, ax):
    """
    Load and preprocess the embryo image.
    
    Parameters:
        image_path (str): Path to the TIFF image
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load image
    img = io.imread(image_path)
    
    # if more than one channels, keep 2nd
    if len(img.shape) > 2:
        img = img[1]

     # Normalize image by dividing by its maximum value
    img_normalized = img / np.max(img)
    
    # Enhance contrast using histogram stretching
    img_enhanced = exposure.rescale_intensity(img_normalized, in_range=(0, 1), out_range=(0, 1))
    
    # Apply Gaussian smoothing to reduce noise
    img_smooth = filters.gaussian(img_enhanced, sigma=10)
    
    ax.imshow(img_smooth)
    
    return [img_normalized,img_smooth]


def detect_embryo_boundary(image, ax):
    """
    Detect the embryo boundary and create a mask & visualization
    
    Parameters:
        image (numpy.ndarray): Preprocessed image
    
    Returns:
        numpy.ndarray: Binary mask of the embryo
    """
    # Otsu thresholding
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    binary = morphology.remove_small_objects(binary, min_size=100)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    if np.sum(binary)==0:
        binary[:]=1
    
    # Find the largest connected component (embryo)
    labels = measure.label(binary)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    embryo_mask = labels == largest_label
     # Visualize steps
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image
#     axes[0,0].imshow(image, cmap='gray')
#     axes[0,0].set_title('Original Image')
    
#     # Binary after thresholding
#     axes[0,1].imshow(binary, cmap='gray')
#     axes[0,1].set_title('After Thresholding')
    
#     # Labeled components
#     axes[1,0].imshow(labels, cmap='nipy_spectral')
#     axes[1,0].set_title('Connected Components')
    
    # Final mask
    ax.imshow(embryo_mask, cmap='gray')
    ax.set_title('Final Embryo Mask')
    
#     plt.tight_layout()
    
    return embryo_mask


def detect_embryo_boundary_alternative(image, low_percentile=10):
    """
    Detect the embryo boundary by excluding the darkest pixels and creating a mask. It's applied 
    when the stripes instead of embryo is segmented when using otsu methods
    
    Parameters:
        image (numpy.ndarray): Preprocessed image
        low_percentile (int): The lower percentile threshold for excluding pixels (default: 5)

    Returns:
        numpy.ndarray: Binary mask of the embryo
    """
    # Exclude the lowest intensity values
    lower_threshold = np.percentile(image, low_percentile)
    filtered_image = image > lower_threshold  # Exclude low values

    # Fill small holes and remove small objects
    binary = morphology.remove_small_objects(filtered_image, min_size=100)
    binary = morphology.remove_small_holes(binary, area_threshold=100)

    # Find the largest connected component (embryo)
    labels = measure.label(binary)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    embryo_mask = labels == largest_label

    # Visualize steps
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')

    # After thresholding
    axes[0, 1].imshow(filtered_image, cmap='gray')
    axes[0, 1].set_title('After Excluding Dark Pixels')

    # Labeled components
    axes[1, 0].imshow(labels, cmap='nipy_spectral')
    axes[1, 0].set_title('Connected Components')

    # Final mask
    axes[1, 1].imshow(embryo_mask, cmap='gray')
    axes[1, 1].set_title('Final Embryo Mask')

    plt.tight_layout()

    return embryo_mask


def generate_ap_profile(img, mask):
    from scipy.ndimage import gaussian_filter1d
    """
    Generate anterior-posterior intensity profile for the specified region,
    normalize the X-axis to percentage of embryo length, and rotate the embryo 
    so that the major axis aligns with the X-axis.
    
    Parameters:
        img (numpy.ndarray): Input image (grayscale)
        mask (numpy.ndarray): Binary mask specifying the region of interest
    
    Returns:
        tuple: (normalized_ap_positions, ap_profile)
            - normalized_ap_positions: Normalized A-P positions (0% to 100%)
            - ap_profile: A-P intensity profile
    """
    img = img*mask
    img[img==0]=None
    
    # Get image dimensions after rotation
    rows, cols = img.shape
    
    # Initialize A-P intensity profile
    ap_profile = np.zeros(cols)
    
    # Calculate intensity profile column by column
    for col in range(cols):
        # Pixels in the current column within the mask
        column_pixels = img[:, col] * mask[:, col]
        
        # Select valid pixels (those within the mask)
        valid_pixels = column_pixels[mask[:, col] == 1]
        
        if valid_pixels.size > 0:  # If there are valid pixels
            ap_profile[col] = np.mean(valid_pixels)
    
    # Normalize x-axis to percentage of embryo length (0 to 100%)
    normalized_ap_positions = np.linspace(0, 100, cols)
    
    ap_profile[ap_profile==0] = None
    
    # Smooth the A-P profile using Gaussian filter
    smoothed_ap_profile = gaussian_filter1d(ap_profile, sigma=20) #was 10
    
    return normalized_ap_positions, smoothed_ap_profile


def find_spots_dog(img, sigma1, sigma2, ax1):
    from scipy.ndimage import gaussian_filter

    gauss1 = gaussian_filter(img, sigma1)
    gauss2 = gaussian_filter(img, sigma2)
    dog = gauss1-gauss2
    thresh = filters.threshold_otsu(dog)
    spotsbin = dog > thresh
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.imshow(dog,vmin = 0, vmax=thresh)
    ax1.set_title('DoG')
#     ax2.imshow(spotsbin, cmap='gray', vmin = 0, vmax = 1)
#     ax2.set_title('thres = '+str(thresh))
#     fig.colorbar(im,ax = ax1)
    return [dog, spotsbin]



def find_stripes_spots(spotsbin):

    contours, hierarchy = cv2.findContours(np.uint8(spotsbin),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     print(contours)
    CentroidX = [0]*len(contours)
    CentroidY = [0]*len(contours)
    i = 0
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        try:
            cX = int(M["m10"] / M["m00"])
        except:
            cX = 0
        try:
            cY = int(M["m01"] / M["m00"])
        except:
            cY = 0

        CentroidX[i] = cX
        CentroidY[i] = cY
        i += 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.scatter(CentroidX, CentroidY)
    ## CLUSTERING CENTROIDS INTO 7 STRIPES
    array = np.dstack((CentroidX,CentroidY))
    arrayCentroids = array[0][~np.all(array[0] == 0, axis=1)]

#     cleaned_array = array[0][~np.isnan(array[0]).any(axis=1)]
    clustering = HDBSCAN().fit(arrayCentroids, eps=1)
#     print(clustering.labels_)
    ax2.scatter(arrayCentroids[:,0],arrayCentroids[:,1],c=clustering.labels_,cmap='viridis')
    return [arrayCentroids]


            

            
def overlay_stripes_spots(spotsbin, Stripes, ax1, ax2):
    import cv2
    contours, hierarchy = cv2.findContours(np.uint8(spotsbin),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     print(contours)
    CentroidX = [0]*len(contours)
    CentroidY = [0]*len(contours)
    i = 0
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        try:
            cX = int(M["m10"] / M["m00"])
        except:
            cX = 0
        try:
            cY = int(M["m01"] / M["m00"])
        except:
            cY = 0

        CentroidX[i] = cX
        CentroidY[i] = cY
        i += 1
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    ax1.scatter(CentroidX, CentroidY)
    ax1.set_aspect('equal')
    ax1.set_ylim(0,2048)
    ax1.set_xlim(0,4096)
    ax1.yaxis.set_inverted(True)
    ## CLUSTERING CENTROIDS INTO 7 STRIPES
    array = np.dstack((CentroidX,CentroidY))
    arrayCentroids = array[0][~np.all(array[0] == 0, axis=1)]
    Labels = np.transpose(Stripes[arrayCentroids[:,1],arrayCentroids[:,0]])
    ax2.scatter(arrayCentroids[:,0],arrayCentroids[:,1],c=Labels,cmap='viridis')
    ax2.set_aspect('equal')
    ax2.set_ylim(0,2048)
    ax2.set_xlim(0,4096)
    ax2.yaxis.set_inverted(True)
    dfSpots = pd.DataFrame({'CentroidX':arrayCentroids[:,0],'CentroidY':arrayCentroids[:,1],'Labels':np.uint8(Labels)})
    return [dfSpots]

def straighten_stripes(dfSpots,ax1,ax2):
    from scipy.optimize import curve_fit
    from scipy.optimize import minimize

    def find_min_distance_point_quadratic(point, a, b, c):
        """
        Calculates the minimum distance between a point and a quadratic function.

        Args:
            point: A tuple (x, y) representing the point.
            a: Coefficient of x^2 in the quadratic function.
            b: Coefficient of x in the quadratic function.
            c: Constant term in the quadratic function.

        Returns:
            The minimum distance between the point and the quadratic function.
        """

        def distance_to_quadratic(x):
            """Calculates the distance from the given point to a point on the quadratic."""
            return np.sqrt((x - point[0])**2 + (a*x**2 + b*x + c - point[1])**2)

        # Use optimization to find the x-value that minimizes the distance
        result = minimize(distance_to_quadratic, x0=0)  # Initial guess x=0
        return [result.fun,result.x]

    def func(x, a, b, c):
#         return a * np.log(b * x) + c
        return a*x**2 + b*x + c
#         return a*x+b
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    dfSpots['minDist'] = np.nan
    dfSpots['adjCentroidX'] = np.nan  
    dfSpots['adjCentroidY'] = np.nan 
    dfSpots['SpotIntensity'] = np.nan
    for i in range(1,8):
#         print(i)
        ## ROTATING X and Y TO FIT TO A PARABOLE, but plotting inverse again
        SingleStripe = dfSpots[dfSpots['Labels']==i]
        popt, pcov = curve_fit(func, SingleStripe['CentroidY'], SingleStripe['CentroidX'])
#         print(popt)
        ax1.plot(SingleStripe['CentroidX'],SingleStripe['CentroidY'] ,'.')
        ax1.plot(func(SingleStripe['CentroidY'], *popt),SingleStripe['CentroidY'], 'g--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         StripeCenter = popt[2]-(popt[1]**2/(4*popt[0]))
        StripeCenter = np.mean(SingleStripe['CentroidX'])
        ax1.vlines(StripeCenter,0,2048, 'k','--')
        ax1.set_aspect('equal')
        ax1.set_ylim(0,2048)
        ax1.set_xlim(0,4096)
        ax1.yaxis.set_inverted(True)
        for spot in range(0,len(SingleStripe)):
            [min_distance,xmin_distance] = find_min_distance_point_quadratic((SingleStripe['CentroidY'].iloc[spot],SingleStripe['CentroidX'].iloc[spot]), popt[0], popt[1], popt[2])
            SingleStripe.loc[SingleStripe.index[spot],'minDist'] = min_distance
            SingleStripe.loc[SingleStripe.index[spot],'adjCentroidX'] = SingleStripe['CentroidX'].iloc[spot]-func(xmin_distance,popt[0], popt[1], popt[2]) + StripeCenter
            SingleStripe.loc[SingleStripe.index[spot],'adjCentroidY'] = SingleStripe['CentroidY'].iloc[spot]
        dfSpots[dfSpots['Labels']==i] = SingleStripe
        ax2.plot(SingleStripe['adjCentroidX'],SingleStripe['adjCentroidY'], '.')
        ax2.vlines(StripeCenter,0,2048, 'k','--')
        ax2.set_aspect('equal')
        ax2.set_ylim(0,2048)
        ax2.set_xlim(0,4096)
        ax2.yaxis.set_inverted(True)
#     print(dfSpots) 
    return (dfSpots)

def puppet_wrap_adjCentroids(image,dfSpots,axs):
    import cv2
    from matplotlib import pyplot as plt
    from pwarp import graph_defined_warp, graph_warp, get_default_puppet
    from pwarp import triangular_mesh

    # image = io.imread(image_path)
#     image = img_enhanced.copy()
    # image = img_smooth.copy()
    image2 = image/np.max(image)*255
    image3 = np.dstack((image2, image2, image2))
    image4 = np.uint8(image3)
    img = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
    Dims = np.shape(img)
    r, f = triangular_mesh(width=Dims[1], height=Dims[0], delta=150)

    #find closest vortex for each centoid
    dfSpots_noNans = dfSpots[dfSpots['Labels']!=0]
    Indexes = [None]*len(dfSpots_noNans)
    for i in range(0,len(dfSpots_noNans)):
        distances = np.linalg.norm(r-(dfSpots_noNans.loc[dfSpots_noNans.index[i],'CentroidX'],dfSpots_noNans.loc[dfSpots_noNans.index[i],'CentroidY']), axis=1)
        min_index = np.argmin(distances)
        Indexes[i] = min_index

    # control points are r points that are going to move, by an amount determined by shift
    control_pts = np.array(Indexes, dtype=int)
    # shifts are new positions, so adjusted centroids
    NewPositions = np.dstack((np.array(dfSpots_noNans.loc[:,'adjCentroidX']),np.array(dfSpots_noNans.loc[:,'adjCentroidY'])))
    shift = np.array(NewPositions[0], dtype=float)
    # puppet = get_default_puppet()
    new_r = graph_warp(
        vertices=r,
        faces=f,
        control_indices=control_pts,
        shifted_locations=shift
    )


    # width, height = 1280, 800
    width, height = Dims[1],Dims[0]
    dx, dy = int(width // 2), int(height // 2)
    dx, dy = 0,0
    scale_x, scale_y = 1, 1
    # r = puppet.r.copy()
    #f = puppet.f
    r[:, 0] = r[:, 0] * scale_x + dx
    r[:, 1] = r[:, 1] * scale_y + dy

    new_r[:, 0] = new_r[:, 0] * scale_x + dx
    new_r[:, 1] = new_r[:, 1] * scale_y + dy

    img_t = graph_defined_warp(
        img,
        vertices_src=r,
        faces_src=f,
        vertices_dst=new_r,
        faces_dst=f
    )
    # default new areas are 255, replace by zeros
    img_t[img_t==255] = 0

#     fig, axs = plt.subplots(1, 3, frameon=False, figsize=(40, 60))
#     plt.tight_layout(pad=0)

    axs[0].imshow(img[:,:,0],vmin=0,vmax=50,cmap='viridis')
    axs[0].triplot(r.T[0], r.T[1], f, lw=0.5)
    axs[1].imshow(img_t[:,:,0],vmin=0,vmax=50,cmap='viridis')
    axs[1].triplot(new_r.T[0], new_r.T[1], f, lw=0.5)
    axs[2].imshow(img_t[:,:,0],vmin=0,vmax=50,cmap='viridis')

    for ax in axs:
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.invert_yaxis()
#         ax.axis('off')
#     plt.show()
    return([img_t[:,:,0]])

def detect_stripes(ap_positions, intensity_profile):
    """
    Detect and segment eve stripes by peak calling algorithm
    
    Parameters:
        ap_positions (numpy.ndarray): A-P axis positions
        intensity_profile (numpy.ndarray): Intensity profile
    
    Returns:
        dict: Dictionary containing stripe boundaries and properties
    """
    # Find peaks (stripe centers)
    peaks, properties = signal.find_peaks(intensity_profile, 
                                        distance=len(ap_positions)/20, prominence = 0.0001) #was 0.006
#     print(peaks)
#     print(properties)
#     print("peaks = ", peaks)
    
    peak_values = intensity_profile[peaks]
    prominences = properties['prominences']
    
    # Sort peaks by their values in descending order
    sorted_peak_indices = np.argsort(prominences)[::-1]
    
    # Get the indices of the top 7 peaks
    top_7_peak_indices = peaks[sorted_peak_indices[:7]]
    
    top_7_peak_indices = top_7_peak_indices[np.argsort(top_7_peak_indices)]
    
    # Get the values of the top N peaks
    top_7_peak_values = intensity_profile[top_7_peak_indices]
#     print(top_7_peak_indices)

    
    stripesDict = {}
    stripesList = [0]*7
    # Identify individual stripes
    for i in range(len(top_7_peak_indices)):
        stripesDict[f'stripe_{i+1}'] = {'center': round(top_7_peak_indices[i]/len(intensity_profile)*100)}
        stripesList[i] = round(top_7_peak_indices[i]/len(intensity_profile)*100)
    
    return [stripesDict,stripesList]

def plot_results(image, ap_positions, intensity_profile, intensity_profile_detrended, stripes, ax1, ax2):
    """
    Create visualization of the analysis results.
    
    Parameters:
        image (numpy.ndarray): Original image
        ap_positions (numpy.ndarray): A-P axis positions
        intensity_profile (numpy.ndarray): Intensity profile
        stripes (dict): Stripe properties
    """
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
    
    # Plot original image
    ax1.imshow(image, cmap='viridis')
    ax1.set_title('Processed Image')
    
    # Plot A-P profile with stripe boundaries
    ax2.plot(ap_positions, intensity_profile, 'gray', label='Intensity Profile')
    ax2.plot(ap_positions, intensity_profile_detrended, 'black', label='Intensity Profile detrended')
#     print(ap_positions, intensity_profile)
    # Highlight the stripes
    for stripe_key, stripe in stripes.items():
    # Plot vertical lines at the center of each stripe
        ax2.axvline(x=stripe['center'], color='indianred', linestyle='--')  

    ax2.set_xlabel('A-P Position (% egg length)', fontsize=12, family='Times New Roman')
    ax2.set_ylabel('Intensity', fontsize=12,family='Times New Roman')
    ax2.set_title('A-P Intensity Profile', fontsize=16, family='Arial')
    ax2.legend()
    
#     plt.tight_layout()
#     return fig



def normalize_intensity(intensity_profile):
    """
    Normalize the intensity profile to range [0, 1] using min-max normalization.
    
    Parameters:
        intensity_profile (numpy.ndarray): Intensity profile to normalize.
    
    Returns:
        numpy.ndarray: Normalized intensity profile.
    """
    intensity_profile = np.array(intensity_profile)
    mask = ~np.isnan(intensity_profile)
    min_intensity = np.nanmin(intensity_profile[mask])
    max_intensity = np.nanmax(intensity_profile[mask])
    intensity_profile_norm = (intensity_profile[mask] - min_intensity) / (max_intensity - min_intensity)
    intensity_profile[mask] = intensity_profile_norm
    return intensity_profile



def detect_stripe_boundaries_afterCrop(ap_profile: np.ndarray, sigma: float = 5, prominence: float = 0.005):

    """
    Detect stripe boundaries based on smoothed intensity profile and peak detection.
    
    Parameters:
        ap_profile (np.ndarray): The original A-P intensity profile.
        sigma (float): Standard deviation for Gaussian smoothing.
        prominence (float): Minimum prominence of peaks to detect stripes.
    
    Returns:
        tuple: (adjusted_ap_profile, start_idx, end_idx)
            - adjusted_ap_profile: Cropped A-P profile within stripe boundaries.
            - start_idx: Index of the left boundary of the first stripe.
            - end_idx: Index of the right boundary of the last stripe.
    """
    # Smooth the profile using a Gaussian filter
    smoothed_profile = gaussian_filter1d(ap_profile, sigma=sigma)
    
    # Detect peaks in the smoothed profile
    peaks, _ = find_peaks(smoothed_profile, prominence=prominence)
    
    if len(peaks) < 2:
        # Adjust to handle cases with fewer peaks
        print("Warning: Less than two peaks detected. Returning full profile.")
        return ap_profile, 0, len(ap_profile) - 1

    # Define the start as the midpoint before the first peak
    start_idx = max(0, peaks[0] - (peaks[1] - peaks[0]) // 2)
    # Define the end as the midpoint after the last peak
    end_idx = min(len(ap_profile) - 1, peaks[-1] + (peaks[-1] - peaks[-2]) // 2)
    
    adjusted_ap_profile = ap_profile[start_idx:end_idx + 1]
    
    print('length of ap_profile is:', len(smoothed_profile))
    print('length of adjusted_ap_profile is:', len( adjusted_ap_profile))
    print('start_idx:', start_idx)
    print('end_idx:', end_idx)
    print('peaks[0] - (peaks[1] - peaks[0]) // 2:', peaks[0] - (peaks[1] - peaks[0]) // 2)
    print('len(ap_profile) - 1:', len(ap_profile) - 1)
    print('peaks[-1] + (peaks[-1] - peaks[-2]) // 2:', peaks[-1] + (peaks[-1] - peaks[-2]) // 2)

    
    return adjusted_ap_profile, start_idx, end_idx


def align_and_plot_profiles(target_profile: np.ndarray, control_profile: np.ndarray, 
                            target_indices: tuple, control_indices: tuple):
    """
    Align and plot A-P profiles of target and control embryos on the same X-axis.
    
    Parameters:
        target_profile (np.ndarray): Cropped A-P profile of the target embryo.
        control_profile (np.ndarray): Cropped A-P profile of the control embryo.
        target_indices (tuple): (start_idx, end_idx) of the target embryo profile.
        control_indices (tuple): (start_idx, end_idx) of the control embryo profile.
    
    Returns:
        None
    """
    # Normalize X-axes to percentage of cropped region
    target_length = len(target_profile)
    control_length = len(control_profile)
    
    target_x = np.linspace(0, 100, target_length)
    control_x = np.linspace(0, 100, control_length)
    
    # Plot the profiles
    plt.figure(figsize=(10, 6))
    plt.plot(target_x, target_profile, label="Target Embryo", color='red', lw=1.5)
    plt.plot(control_x, control_profile, label="Control Embryo", color="blue", lw=1.5, linestyle="--")
    plt.xlabel("Pixel position",fontsize=12, family='Times New Roman')
    plt.ylabel("Intensity",fontsize=12, family='Times New Roman')
    plt.title("Aligned A-P Profiles of Target and Control Embryos_cropped",fontsize=16, family='Arial')
    plt.legend()
    plt.show()
