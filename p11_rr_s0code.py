import pandas as pd  # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation # type: ignore
import cv2
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

df = pd.read_csv('P11_RR_S0_cricketball.csv')
dfs1 = pd.read_csv('P11RR_S1_post1.csv')
# print(df.head(5))
df.columns = df.columns.str.strip()
dfs1.columns = dfs1.columns.str.strip()
dfs1['Z7'] = (dfs1['Z7'] - dfs1['Z7'].max()) * -1
df.dropna(axis=1)
# print(df['Sensor2'])

# Sensor 2: Index middle (micro)
# Sensor 3: Index proximal (micro)
# Sensor 4: Thumb distal (micro)
# Sensor 5: Thumb middle (micro)
# Sensor 6: Index distal (micro)
# Sensor 7: Palm (macro)
# Sensor 8: Lower arm (macro) - able to calculate sup/prono raw 
# Sensor 9: Upper arm (macro)- flexion and abduction can be calculated raw
# Sensor 10: Object (macro)
#plot range of motion(y) for every sensor(1-6) of patient 11 RR during S0

#plotting object 
plt.figure(figsize=(12,6))
plt.plot(dfs1['X7 (IN)'])
plt.grid(True)
plt.legend()
plt.show()


    

#calculate rotation vector given 
stationary_vector = [1,0,0]
def rotation_vector(el_angle,az_angle,r_angle):
    # Convert degrees to radians
    e = np.radians(el_angle)  # elevation y 
    a = np.radians(az_angle)  # azimuth z
    r = np.radians(r_angle)  # roll x 

    # trig terms
    ce = np.cos(e)
    se = np.sin(e)
    ca = np.cos(a)
    sa = np.sin(a)
    cr = np.cos(r)
    sr = np.sin(r)

    # Rotation matrix R(e, a, r)
    R = np.array([
        [ce * ca, ce * sa, -se],
        [-cr * sa + sr * se * ca, cr * ca + sr * se * sa, sr * ce],
        [sr * sa + cr * se * ca, -sr * ca + cr * se * sa, cr * ce]
    ])

    return R @ stationary_vector

#find angle of relative rotation via trace
def vector_angle(vec1, vec2):
    # Normalize vectors just in case
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)  # clip for numerical stability
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


#sensor 2 index middle
azimuth_sensor2 = df['Az2 (DEG)'].mean()
elevation_sensor2 = df['El2'].mean()
roll_sensor2 = df['Rl2'].mean()
sensor2_rotation = rotation_vector(elevation_sensor2,azimuth_sensor2,roll_sensor2)

#sensor 4 thumb distal 
azimuth_sensor4 = df['Az4 (DEG)'].mean()
elevation_sensor4 = df['El4'].mean()
roll_sensor4 = df['Rl4'].mean()
sensor4_rotation = rotation_vector(elevation_sensor4,azimuth_sensor4,roll_sensor4)

#sensor 8 lower arm
azimuth_sensor8 = df['Az8 (DEG)'].mean()
elevation_sensor8 = df['El8'].mean()
roll_sensor8 = df['Rl8'].mean()
sensor8_rotation = rotation_vector(elevation_sensor8,azimuth_sensor8,roll_sensor8).round(2)

#sensor 9 upper arm
azimuth_sensor9 = df['Az9 (DEG)'].mean()
elevation_sensor9 = df['El9'].mean()
roll_sensor9 = df['Rl9'].mean()
sensor9_rotation = rotation_vector(elevation_sensor9,azimuth_sensor9,roll_sensor9).round(2)

elbow_flexion = np.dot(sensor8_rotation,sensor9_rotation).round(2) #computed using dot product method as outlined in Khanna paper


elbow_flexion_angles = [] #baseline correct, start point is 0 degrees, and record change from there 
#elbow flexion visualization 
for i in range(len(df)):
    # Sensor 8 (forearm) rotation
    azimuth_sensor8 = df['Az8 (DEG)'][i]
    elevation_sensor8 = df['El8'][i]
    roll_sensor8 = df['Rl8'][i]
    sensor8_rotation = rotation_vector(elevation_sensor8, azimuth_sensor8, roll_sensor8)
    
    # Sensor 9 (upper arm) rotation
    azimuth_sensor9 = df['Az9 (DEG)'][i]
    elevation_sensor9 = df['El9'][i]
    roll_sensor9 = df['Rl9'][i]
    sensor9_rotation = rotation_vector(elevation_sensor9, azimuth_sensor9, roll_sensor9)
    
    # Calculate elbow flexion using the dot product
    elbow_flexion = np.dot(sensor8_rotation, sensor9_rotation)
    elbow_flexion_angles.append(elbow_flexion)
df['elbow_flexion'] = elbow_flexion_angles
df['elbow_flexion_deg'] = np.degrees(np.arccos(np.clip(df['elbow_flexion'], -1.0, 1.0)))
df['elbow_flexion_deg'] = df['elbow_flexion_deg'] - df['elbow_flexion_deg'].iloc[0]  # Normalize to start at zero

# baseline corrected 
plt.figure(figsize=(12,8))
plt.plot(df['elbow_flexion_deg'], label='Elbow Flexion (baseline corrected )', color='darkorange')

# Overlay horizontal lines at 60°, 90°, 120°, 150° flexion
for angle in [60, 90, 120, 150]:
    plt.axhline(angle, color='red', linestyle='--', alpha=0.6, label=f'{angle}° flexion')
plt.ylabel('Degrees')

plt.title('Elbow Flexion of Patient 11RR During S0')
plt.grid(True)
plt.legend()
plt.show()



#sensor 10 rotation object 
azimuth_sensor10 = df['Az10 (DEG)'].mean()
elevation_sensor10 = df['El10'].mean()
roll_sensor10 = df['Rl10'].mean()
sensor10_rotation = rotation_vector(elevation_sensor10,azimuth_sensor10,roll_sensor10).round(2)

#calculate angles between forearm and object by multipying inverse of objects rotation matrix 
rel_rotation = sensor9_rotation @ sensor10_rotation #relative rotation from shoulder to object


#finding 0,30,60 degree angles for data 
shoulder_angles = []
for i in range(len(df)):
    r_obj = rotation_vector(df['El10'][i], df['Az10 (DEG)'][i], df['Rl10'][i])
    r_shoulder = rotation_vector(df['El9'][i],df['Az9 (DEG)'][i],df['Rl9'][i])
    angle = vector_angle(r_obj, r_shoulder)
    shoulder_angles.append(angle)
df['shoulder_angles'] = shoulder_angles
df['shoulder_angles'] = df['shoulder_angles'] - df['shoulder_angles'].iloc[0]
    
    # find points where angle is -30, 0,30,60 for shoulder flexion/abduction 
negthirty_vals = df['shoulder_angles'][np.isclose(df['shoulder_angles'], -30, atol=2)]
zero_vals      = df['shoulder_angles'][np.isclose(df['shoulder_angles'], 0, atol=2)]
thirty_vals    = df['shoulder_angles'][np.isclose(df['shoulder_angles'], 30, atol=2)]
sixty_vals     = df['shoulder_angles'][np.isclose(df['shoulder_angles'], 60, atol=2)]

#shoulder flexion visualization 
plt.figure(figsize=(12,6))
plt.plot(df['shoulder_angles'], label = 'Shoulder Flexion Angle', color = 'royalblue')
for angle in [0,30,60]:
    plt.axhline(angle, color = 'purple', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.ylim(0,150)    
plt.ylabel('Shoulder Angle (Degrees)')
plt.title('Shoulder Flexion of Patient 11 RR During S0')
plt.legend()
plt.grid(True)
plt.show()

#shoulder abduction visualization 
plt.figure(figsize=(12,6))
plt.plot(df['shoulder_angles'], label = 'Shoulder Abduction Angle', color = 'orange')
for angle in [-30,0,30]:
    plt.axhline(angle, color = 'green', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.ylim(-50,150)    
plt.ylabel('Shoulder Angle (Degrees)')
plt.title('Shoulder Abduction of Patient 11 RR During S0')
plt.legend()
plt.grid(True)
plt.show()

#visualization of occurences for shoulder flexion 
counts = [
    len(zero_vals),
    len(thirty_vals),
    len(sixty_vals)
]

labels = [ '~0°', '~30°', '~60°']
colors = ['#FF6B6B', '#6BCB77', '#4D96FF', '#FFD93D']

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=colors)

plt.title('Frequency of Shoulder Flexion Near Key Angles')
plt.xlabel('Target Angle Range')
plt.ylabel('Number of Occurrences')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#palm calculations and visualization 
palm_angles = []
for i in range(len(df)):
    r_obj = rotation_vector(df['El10'][i], df['Az10 (DEG)'][i], df['Rl10'][i])
    r_palm = rotation_vector(df['El7'][i], df['Az7 (DEG)'][i],df['Rl7'][i])
    angle = vector_angle(r_obj, r_palm)
    palm_angles.append(angle)
df['palm_angles'] = palm_angles
df['palm_angles'] = df['palm_angles'] - df['palm_angles'].iloc[0]
plt.figure(figsize = (12,6))
plt.plot(df['palm_angles'], label = 'Forearm Supination/Pronation', color = 'maroon')
for angle in [0,90]:
    plt.axhline(angle, color = 'cyan', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.ylabel('Palm Angles (Degrees)')
plt.title('Forearm Supination and Pronoation of Patient 11RR Throughout S0')
plt.legend()
plt.grid(True)
plt.show()

#cumulative visualization of shoulder flexion,abduction, and elbow flexion 
plt.figure(figsize=(12, 8))

# Plot shoulder flexion
plt.plot(df.index, df['shoulder_angles'], label='Shoulder Flexion', color='blue')

# Plot forearm subination
plt.plot(df.index, df['palm_angles'], label='Palm Supination', color='maroon')

# Plot elbow flexion (converted from dot product to angle)
elbow_flexion_deg = np.degrees(np.arccos(np.clip(df['elbow_flexion'], -1, 1)))
plt.plot(df.index, elbow_flexion_deg, label='Elbow Flexion', color='darkorange')

# Shoulder abduction angle — you'll need to calculate it similar to how you did for flexion
# Placeholder: replace with actual abduction angle array
# For now, let's assume you have something like df['shoulder_abduction']
# plt.plot(df.index, df['shoulder_abduction'], label='Shoulder Abduction', color='green')

plt.ylabel('Joint Angle (Degrees)')
plt.xlabel('Frame Index')
plt.title('Joint ROM Angles of Patient 11 RR During S0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def detect_and_plot_peaks(objectsignal, palmsignal, num_peaks=6, num_min = 15, polyorder=3, window_length=11,
                          smoothing=True, epsilon=0.01):
    """
    Input: series for the object signal and palm signal 
    Detect peaks in objectsignal, use palmsignal to find start/end of each trial.
    Plot two dotted black lines per trial (start and end), and red dots for peaks.
    Output: Graph with palmsignal and objectsignal plotted along with peaks in the object signal 
    & start and end times 
    """
    #finding baseline of palm based on lowest in y value occuring mode 

    
    # Normalize object signal
    objectsignal = (objectsignal - objectsignal.max()) * -1
    objectsignal = (objectsignal - np.min(objectsignal)) / (np.max(objectsignal) - np.min(objectsignal))
    objectsignal = objectsignal * 10

    if smoothing:
        smoothed_signal = savgol_filter(objectsignal, window_length=window_length, polyorder=polyorder)
        smoothed_signal -= smoothed_signal.max()
    else:
        smoothed_signal = objectsignal

    # Detect peaks & minima
    all_peaks, properties = find_peaks(smoothed_signal, prominence=0)
    minima_indices, _ = find_peaks(-palmsignal, distance=5, prominence=0.5)

    #taking most prominent maxima  
    if len(all_peaks) > num_peaks:
        prominences = properties['prominences']
        top_indices = np.argsort(prominences)[-num_peaks:]
        peaks = all_peaks[top_indices]
        peaks = peaks[np.argsort(peaks)]
    else:
        peaks = all_peaks

    #eliminating non trials 

    min_indices = []
    for peak in peaks:
        rightminima = minima_indices[minima_indices > peak]

        if len(rightminima) < 1:
            continue
        else:
            distances = np.abs(rightminima-peak) #assume peaks arent sorted
            closestindices = np.argsort(distances)[:3] #3 closest xvals to palm peak
            closestx = rightminima[closestindices] #closest y values 
            lowestmin = closestx[np.argmin(palmsignal[closestx])]
            min_indices.append(lowestmin)
    
    # Palm signal baseline, hardcode to be 
    palmbaseline = dfs1['X7 (IN)'].quantile(0.25)
 
    #determine where the lift is for respective peak and denote landing as return to respective peak
    lift_offs = []
    returns = []

    for peak, minima in zip(peaks, min_indices):
        # Left boundary (lift-off)
        left = peak
        while left > 0 and palmsignal[left] > palmbaseline + epsilon:
            left -= 1
        lift_offs.append(left)

    def is_clear_peak(signal, start, stop, min_prominence=0.5, min_width=5):

        segment = signal[start:stop]

        # Find all peaks in this segment
        peaks, properties = find_peaks(segment, prominence=min_prominence, width=min_width)

        if len(peaks) > 0:
            return True  # At least one clear peak
        else:
            return False  # No peak or only noise



        # Save lift-off value for return threshold
        # lift_val = palmsignal[left]

        # # Right boundary (return to lift-off value)
        # right = peak
        # while right < len(palmsignal) - 1 and palmsignal[right] < (palmsignal[minima]-epsilon):
        #     right += 1
        # returns.append(right)


    # Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(objectsignal, label='Object Signal')
    plt.plot(palmsignal, label='Palm Signal')
    plt.scatter(peaks, objectsignal[peaks], color='red', label='Detected Peaks')
    # plt.scatter(min_indices, palmsignal[min_indices], color='green', label='palm Peaks')
    for i, (lo, re) in enumerate(zip(lift_offs, min_indices)):
        plt.axvline(x=lo, color='k', linestyle='--', label='Start' if i == 0 else "")
        plt.axvline(x=re, color='m', linestyle='--', label='End' if i == 0 else "")
    plt.legend()
    plt.title("Trials: Peaks from Object Signal, Start/End from Palm Signal")
    plt.xlabel("Time Index")
    plt.ylabel("Normalized Signal")
    plt.show()

    return peaks, lift_offs, min_indices
#returns 3 lists 
    



print(detect_and_plot_peaks(dfs1['Z10'], dfs1['X7 (IN)']))

# def find_return_to_ground_peaks(signal, baseline=None, min_distance=5, plot=True, title='Peaks Baseline Defined by Groundlevel'):
#     """
#     Detects peaks as the maximum between when a signal leaves 'ground' and returns.
    
#     Parameters:
#     - signal: 1D array-like data.
#     - baseline: Value representing the 'ground'. If None, uses the first value.
#     - min_distance: Minimum samples between a new peak (for noise filtering).
#     - plot: Whether to plot the signal with the peaks and regions.
#     - title: Title of the plot.

#     Returns:
#     - peaks_indices: Indices of the identified peaks.
#     """

#     signal = np.asarray(signal)
#     baseline = signal[0] if baseline is None else baseline

#     above_ground = signal > baseline
#     peaks_indices = []
#     i = 0

#     while i < len(signal):
#         # look for lift 
#         if above_ground[i]:
#             start = i
#             # move forward until returns to ground
#             while i < len(signal) and above_ground[i]:
#                 i += 1
#             end = i

#             # find the peak in this segment
#             if end - start > 0:
#                 segment = signal[start:end]
#                 peak_offset = np.argmax(segment)
#                 peak_idx = start + peak_offset

#                 # Optional: skip tiny peaks (noise) by checking distance
#                 if len(peaks_indices) == 0 or (peak_idx - peaks_indices[-1]) > min_distance:
#                     peaks_indices.append(peak_idx)
#         else:
#             i += 1

#     if plot:
#         plt.figure(figsize=(12, 6))
#         plt.plot(signal, label='Signal', color='blue')
#         plt.axhline(baseline, color='green', linestyle='--', label='Ground Level')

#         # Mark peaks
#         plt.scatter(peaks_indices, signal[peaks_indices], color='red', zorder=5, label='Detected Peaks')
        

#         # Draw dotted lines before & after peak regions
#         for idx in peaks_indices:
#             # Find region edges
#             left = idx
#             right = idx
#             while left > 0 and signal[left] > baseline:
#                 left -= 1
#             while right < len(signal) - 1 and signal[right] > baseline:
#                 right += 1
#             plt.axvline(left, linestyle=':', color='black', alpha=0.6)
#             plt.axvline(right, linestyle=':', color='black', alpha=0.6)

#         plt.title(title)
#         plt.xlabel('Sample Index')
#         plt.ylabel('Signal Value')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

#     return peaks_indices
# find_return_to_ground_peaks(dfs1['Z10'])

#plane method
XY_plane = np.array([[1, 0, 0], [0, 1, 0]]).T  # shape (3, 2)
XZ_plane = np.array([[1, 0, 0], [0, 0, 1]]).T

def compute_plane_angle_from_rotvecs(
    rotvec1, rotvec2,         # 3x1 rotation vectors for sensor1 and sensor2
    roll1_base, roll2_base,   # baseline roll values (in radians)
    roll1, roll2,             # current roll values (in radians)
    plane_type='XY'           # 'XY' or 'XZ' plane
):
    """
    Computes signed bend angle between two planes defined by sensor rotations.

    Parameters:
    - rotvec1, rotvec2: 3x1 rotation vectors for each sensor.
    - roll1_base, roll2_base: baseline roll values (in radians).
    - roll1, roll2: current roll values (in radians).
    - plane_type: 'XY' (default) or 'XZ'.

    Returns:
    - theta: signed angle (in radians).
    """

    # convert rotation vectors (3x1) to rotation matrices (3x3)
    R1, _ = cv2.Rodrigues(rotvec1)
    R2, _ = cv2.Rodrigues(rotvec2)

    # compute relative roll
    rel_r1 = roll1 - roll1_base
    rel_r2 = roll2 - roll2_base

    # define initial plane
    if plane_type.upper() == 'XY':
        plane = np.array([[1, 0],
                          [0, 1],
                          [0, 0]])
        sign_axis = 1
    elif plane_type.upper() == 'XZ':
        plane = np.array([[1, 0],
                          [0, 0],
                          [0, 1]])
        sign_axis = 2
    else:
        raise ValueError("plane_type must be 'XY' or 'XZ'")

    def apply_relative_roll(R, rel_roll):
        roll_mat = np.array([
            [np.cos(rel_roll), -np.sin(rel_roll), 0],
            [np.sin(rel_roll),  np.cos(rel_roll), 0],
            [0,                 0,                1]
        ])
        return R @ roll_mat

    R1_full = apply_relative_roll(R1, rel_r1)
    R2_full = apply_relative_roll(R2, rel_r2)

    # rotate plane basis
    plane1_rot = R1_full @ plane
    plane2_rot = R2_full @ plane

    # normal to plane 1
    normal1 = np.cross(plane1_rot[:, 0], plane1_rot[:, 1])
    normal1 /= np.linalg.norm(normal1)

    # project plane2 vector onto plane1
    vec2 = plane2_rot[:, 0]
    projection = vec2 - np.dot(vec2, normal1) * normal1

    # compute angle
    numerator = np.dot(projection, vec2)
    denominator = np.linalg.norm(projection) * np.linalg.norm(vec2)
    angle = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))

    # sign from cross product
    cross = np.cross(projection, vec2)
    cross_local = np.linalg.inv(R2_full) @ cross
    if cross_local[sign_axis] > 0:
        angle *= -1

    return angle

#index mcp calculated using sensors 3 and 7
indexmcp_angles = []
for i in range(len(df)):
    palm_rm = rotation_vector(df['El7'].iloc[i], df['Az7 (DEG)'].iloc[i], df['Rl7'].iloc[i])
    indexprox_rm = rotation_vector(df['El3'].iloc[i], df['Az3 (DEG)'].iloc[i], df['Rl3'].iloc[i])
    palm_br = np.radians(df['Rl7'][0])
    indexprox_br = np.radians(df['Rl3'][0])
    palm_cr = np.radians(df['Rl7'].mean())
    indexprox_cr = np.radians(df['Rl3'].mean())
    thumb_mcp = compute_plane_angle_from_rotvecs(palm_rm,indexprox_rm, palm_br, indexprox_br, palm_cr, indexprox_cr, 'XY')
    indexmcp_angles.append(thumb_mcp)
df['indexmcp_angles'] = indexmcp_angles
plt.figure(figsize=(12,6))
plt.plot(np.abs(np.degrees(df['indexmcp_angles'])), color = 'orange', label = 'Index MP Angles')
for angle in [0,30, 60]:
    plt.axhline(angle, color = 'blue', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.title('Index MP Angles of Patient11RR During S0')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.legend()
plt.show()

#index pip calculated using sensors 2 & 3
indexpip_angles = []
for i in range(len(df)):
    indexmid_rm = rotation_vector(df['El2'].iloc[i], df['Az2 (DEG)'].iloc[i], df['Rl2'].iloc[i]) #sensor 2
    indexprox_rm = rotation_vector(df['El3'].iloc[i], df['Az3 (DEG)'].iloc[i], df['Rl3'].iloc[i])
    indexmid_br = np.radians(df['Rl2'][0])
    indexprox_br = np.radians(df['Rl3'][0])
    indexmid_cr = np.radians(df['Rl2'].mean())
    indexprox_cr = np.radians(df['Rl3'].mean())
    indexpip = compute_plane_angle_from_rotvecs(indexmid_rm,indexprox_rm, indexmid_br, indexprox_br, indexmid_cr, indexprox_cr, 'XY')
    indexpip_angles.append(indexpip)
df['indexpip_angles'] = indexpip_angles
plt.figure(figsize=(12,6))
plt.plot(np.abs(np.degrees(df['indexpip_angles'])), color = 'purple', label = 'Index PIP Angles')
for angle in [0,30,60]:
    plt.axhline(angle, color = 'green', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.title('Index PIP Angles of Patient11RR During S0')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.legend()
plt.show()

#calculating index dip using sensors 2 & 4
indexdip_angles = []
for i in range(len(df)):
    indexmid_rm = rotation_vector(df['El2'].iloc[i], df['Az2 (DEG)'].iloc[i], df['Rl2'].iloc[i]) #sensor 2
    thumbdistal_rm = rotation_vector(df['El4'].iloc[i], df['Az4 (DEG)'].iloc[i], df['Rl4'].iloc[i])
    indexmid_br = np.radians(df['Rl2'][0])
    thumbdistal_br = np.radians(df['Rl4'][0])
    indexmid_cr = np.radians(df['Rl2'].mean())
    thumbdistal_cr = np.radians(df['Rl4'].mean())
    indexdip = compute_plane_angle_from_rotvecs(indexmid_rm,thumbdistal_rm, indexmid_br, thumbdistal_br, indexmid_cr, thumbdistal_cr, 'XY')
    indexdip_angles.append(indexdip)
df['indexdip_angles'] = indexdip_angles
plt.figure(figsize=(12,6))
plt.plot(np.abs(np.degrees(df['indexdip_angles'])), color = 'green', label = 'Index DIP Angles')
for angle in [0,20,50]:
    plt.axhline(angle, color = 'orange', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.title('Index DIP Angles of Patient11RR During S0')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.legend()
plt.show()

#calculating wrist extension using sensors 7 & 8 
wristextension_angles = []
for i in range(len(df)):
    palm_rm = rotation_vector(df['El7'].iloc[i], df['Az7 (DEG)'].iloc[i], df['Rl7'].iloc[i]) 
    forearm_rm = rotation_vector(df['El8'].iloc[i], df['Az8 (DEG)'].iloc[i], df['Rl8'].iloc[i])
    palm_br = np.radians(df['Rl7'][0])
    forearm_br = np.radians(df['Rl8'][0])
    palm_cr = np.radians(df['Rl7'].mean())
    forearm_cr = np.radians(df['Rl8'].mean())
    wristextension = compute_plane_angle_from_rotvecs(palm_rm,forearm_rm,palm_br,forearm_br,palm_cr,forearm_cr, 'XY')
    wristextension_angles.append(wristextension)
df['wristextension_angles'] = wristextension_angles
plt.figure(figsize=(12,6))
plt.plot(np.abs(np.degrees(df['wristextension_angles'])), color = 'grey', label = 'Wrist Extension Angles')
for angle in [0,10,40]:
    plt.axhline(angle, color = 'blue', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.title('Wrist Extension/Flexion Angles of Patient11RR During S0')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.legend()
plt.show()

#calculating wrist abduction using sensors 7 & 8 but xz plane 
wristabduction_angles = []
for i in range(len(df)):
    palm_rm = rotation_vector(df['El7'].iloc[i], df['Az7 (DEG)'].iloc[i], df['Rl7'].iloc[i]) 
    forearm_rm = rotation_vector(df['El8'].iloc[i], df['Az8 (DEG)'].iloc[i], df['Rl8'].iloc[i])
    palm_br = np.radians(df['Rl7'][0])
    forearm_br = np.radians(df['Rl8'][0])
    palm_cr = np.radians(df['Rl7'].mean())
    forearm_cr = np.radians(df['Rl8'].mean())
    wristabduction = compute_plane_angle_from_rotvecs(palm_rm,forearm_rm,palm_br,forearm_br,palm_cr,forearm_cr, 'XZ')
    wristabduction_angles.append(wristabduction)
df['wristabduction_angles'] = wristabduction_angles
plt.figure(figsize=(12,6))
plt.plot(np.abs(np.degrees(df['wristabduction_angles'])), color = 'orange', label = 'Wrist Abduction Angles')
for angle in [-20,0,10]:
    plt.axhline(angle, color = 'purple', linestyle = '--', alpha = 0.6, label = f'{angle}°')
plt.title('Wrist Abduction Angles of Patient11RR During S0')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.legend()
plt.show()

#cumulative visualization 
plt.figure(figsize=(12, 8))

# Plot shoulder flexion
plt.plot(df.index, df['shoulder_angles'], label='Shoulder Flexion', color='red')

# Plot forearm subination
plt.plot(df.index, df['palm_angles'], label='Palm Supination', color='maroon')

# Plot elbow flexion (converted from dot product to angle)
elbow_flexion_deg = np.degrees(np.arccos(np.clip(df['elbow_flexion'], -1, 1)))
plt.plot(df.index, elbow_flexion_deg, label='Elbow Flexion', color='darkorange')

plt.plot(np.abs(np.degrees(df['indexmcp_angles'])), color = 'orange', label = 'Thump MP Angles')

plt.plot(np.abs(np.degrees(df['indexpip_angles'])), color = 'blue', label = 'Index PIP Angles')

plt.plot(np.abs(np.degrees(df['indexdip_angles'])), color = 'green', label = 'Index DIP Angles')

plt.plot(np.abs(np.degrees(df['wristextension_angles'])), color = 'teal', label = 'Wrist Extension Angles')

plt.plot(np.abs(np.degrees(df['wristabduction_angles'])), color = 'pink', label = 'Wrist Abduction Angles')
# Shoulder abduction angle — you'll need to calculate it similar to how you did for flexion
# Placeholder: replace with actual abduction angle array
# For now, let's assume you have something like df['shoulder_abduction']
# plt.plot(df.index, df['shoulder_abduction'], label='Shoulder Abduction', color='green')

plt.ylabel('Joint Angle (Degrees)')
plt.xlabel('Frame Index')
plt.title('Joint ROM Angles of Patient 11 RR During S0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#polar plot good for showing one angle on a circular plane 
theta = np.deg2rad(df['shoulder_angles'])
r = range(len(theta))
plt.polar(theta, r)
plt.title('Shoulder Angle in Polar Coordinates')
plt.show()

#new df for joint angles segmented by trials with peak-to-peak amplitude calculated for change, ROM & T2TV included
joint_angles_df = pd.DataFrame({
    'elbow_flexion_deg': df['elbow_flexion_deg'],
    'indexdip_angles': np.degrees(df['indexdip_angles']),
    'indexmcp_angles': np.degrees(df['indexmcp_angles']),
    'indexpip_angles': np.degrees(df['indexpip_angles']),
    'palm_angles': np.degrees(df['palm_angles']),
    'shoulder_angles': df['shoulder_angles'],
    'wristabduction_angles': np.degrees(df['wristabduction_angles']),
    'wristextension_angles': np.degrees(df['wristextension_angles'])
})

#add peak-to-peak amplitude per trial
# def change_per_trial(df, num_peaks=6, min_distance=5, tolerance = 0.01, min_seperation = 200):
#     all_peaks = {}
#     signal = np.asarray(df[col])
#     #calculate trials 
#     # for col in df.columns:
#     #     signal = np.asarray(df[col])
#     #     baseline = signal[0]

#     #     #  signal rises above baseline
#     #     above_ground = signal > baseline
#     #     peaks_indices = []
#     #     i = 0

#     #     while i < len(signal):
#     #         if above_ground[i]:
#     #             start = i
#     #             while i < len(signal) and above_ground[i]:
#     #                 i += 1
#     #             end = i

#     #             if end - start > 0:
#     #                 segment = signal[start:end]
#     #                 peak_offset = np.argmax(segment)
#     #                 peak_idx = start + peak_offset

#     #                 if len(peaks_indices) == 0 or (peak_idx - peaks_indices[-1]) > min_distance:
#     #                     peaks_indices.append(peak_idx)
#     #             else:
#     #                 i += 1
#     #         else:
#     #             i += 1

#     #     #  N most prominent peaks
#     #     if len(peaks_indices) > num_peaks:
#     #         prominences = signal[peaks_indices] - baseline
#     #         top_indices = np.argsort(prominences)[-num_peaks:]
#     #         peaks_indices = [peaks_indices[j] for j in sorted(top_indices)]

#     #     all_peaks[col] = peaks_indices

#     all_peaks = find_peaks(signal)


#     maxheight = signal[all_peaks].max()

#     #peaks within 10%  tolerance, eliminate peaks within a given horizontal range 
#     selectedpeaks = [x for x in all_peaks if signal[x] >= (1-tolerance) * maxheight]
#     filtered_peaks = [p for i, p in enumerate(sorted(selectedpeaks)) 
#                   if i == 0 or abs(p - sorted(selectedpeaks)[i-1]) >= min_seperation]

#     selectedpeaks = np.array(filtered_peaks)

#         # columns for trial changes
#         for trial_num in range(1, num_peaks):  # num_peaks trials
#             #  start and end of the trial
#             start_idx = peaks_indices[trial_num - 1]
#             end_idx = peaks_indices[trial_num]

#             # values at start and end of the trial
#             start_value = signal[start_idx]
#             end_value = signal[end_idx]
#             delta = abs(start_value-end_value)
#             print(f"Trial {trial_num}: Delta between {start_idx} and {end_idx}: {delta}")

#             # Create a new column for each trial's delta
#             delta_col = f"{col}_trial_{trial_num}_delta"
#             # Assign delta value to the column for the trial
#             df[delta_col] = delta

#     return df, all_peaks


change_per_trial(joint_angles_df)
print(joint_angles_df.head(6))
#calculating ROM test 

