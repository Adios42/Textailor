import numpy as np

def interpolate(a, b, ratio=0.5):
    """Interpolate between two points a and b given a ratio."""
    # Check if either endpoint has extreme elevation values
    azim_a,azim_b=a[1],b[1]
    if abs(a[0]) == 90:
        # Use the azimuth from the other point and interpolate only the elevation
        interpolated_azim = azim_b
    elif abs(b[0]) == 90:
        # Use the azimuth from the other point and interpolate only the elevation
        interpolated_azim = azim_a
    else:
        # Calculate the difference
        delta = azim_b - azim_a
        # Correct for wrap-around in the positive direction
        if delta > 180:
            azim_a = azim_a+360  # Add 360 to start angle to correct path
        # Correct for wrap-around in the negative direction
        elif delta < -180:
            azim_b =azim_b+360  # Add 360 to end angle to correct path

        interpolated_azim=(azim_a + ratio * (azim_b - azim_a))%360

    return (a[0] + ratio * (b[0] - a[0])), interpolated_azim

def interpolate_first_pair_repeatedly(pairs, repetitions=1):
    def interpolate_once(pairs):
        if len(pairs) < 2:
            return pairs  # If less than two elements, no interpolation needed.
        # Interpolate between the last two elements
        last_pair = pairs[-1]
        second_last_pair = pairs[-2]
        interpolated_point = interpolate(second_last_pair, last_pair)
        # Insert the interpolated point right before the last element
        new_pairs = pairs[:-1] + [interpolated_point, last_pair]
        return new_pairs

    result = pairs
    for _ in range(repetitions):
        result = interpolate_once(result)
    return result

def find_past_view(current_elev,current_azim,past_elev,past_azim):

    # Initialization of variables for finding the nearest point
    min_distance = float('inf')
    closest_indices = []

    # Calculate the distance to all previous viewpoints
    for i in range(len(past_elev)):
        azim_diff = min(abs(past_azim[i] - current_azim), 360 - abs(past_azim[i] - current_azim))
        elev_diff = abs(past_elev[i] - current_elev)
        
        # Distance calculation considering both elevation and azimuth differences
        distance = np.sqrt(azim_diff**2 + elev_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_indices = [i]  # Initialize the index of the smallest distance
        elif distance == min_distance:
            closest_indices.append(i)  # "If the distance is the same, append the index

    #Find the point with azimuth equal to 0 among the nearest points
    for index in closest_indices:
        if past_azim[index] == 0:
            closest_index = index
            break
    else:
        # If there is no point with azimuth equal to 0, select the first nearest point
        closest_index = closest_indices[0]

    return past_elev[closest_index],past_azim[closest_index]

