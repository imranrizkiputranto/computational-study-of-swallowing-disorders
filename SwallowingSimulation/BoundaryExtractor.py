import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import sys
from tqdm import tqdm
import matplotlib
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Polygon, LineString, MultiPolygon
from numba import njit
matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

directory = 'D:/Things/Programming/IRP/WCSPH/Numba/InterImages/reconstructed_image_'
directory1 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Initial/boundary'
directory2 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Ghost/boundary'
directory3 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Velocity_initial/vel'
directory4 = 'D:/Things/Programming/IRP/WCSPH/Numba/npyfiles/Swallowing/BoundariesFinal/Velocity_ghost/vel'
length = 2400
dt = 0.00017857


def resample_contour(contour, num_points):
    contour = contour[:, 0, :]  # Remove extra dimension
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_distances = np.hstack(([0], np.cumsum(distances)))
    total_distance = cumulative_distances[-1]
    target_distances = np.linspace(0, total_distance, num_points)

    new_points = []
    for target_distance in target_distances:
        index = np.searchsorted(cumulative_distances, target_distance)
        if index == 0:
            new_point = contour[0]
        elif index == len(contour):
            new_point = contour[-1]
        else:
            weight = (target_distance - cumulative_distances[index - 1]) / distances[index - 1]
            new_point = (1 - weight) * contour[index - 1] + weight * contour[index]
        new_points.append(new_point)

    return np.array(new_points)


def find_centroid(points):
    return np.mean(points, axis=0)


def rotate(points, angle, center):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    points_rotated = np.dot(points - center, rotation_matrix) + center
    return points_rotated


def opencv_contour_area(points):
    contour = points.astype(np.int32)  # Convert the points to the required data type
    area = cv2.contourArea(contour)
    return area


def cut_bottom_part(points, cut_y, cut_x):
    remaining_points_y = points[points[:, 1] >= cut_y]
    cut_points_y = points[points[:, 1] < cut_y]

    cut_left = cut_points_y[cut_points_y[:, 0] < cut_x]
    cut_right = cut_points_y[cut_points_y[:, 0] >= cut_x]

    return remaining_points_y, cut_left, cut_right


def find_closest_points(points, y_value, num_points=6):
    y_diffs = np.abs(points[:, 1] - y_value)
    closest_indices = np.argpartition(y_diffs, num_points)[:num_points]
    closest_points = points[closest_indices]
    return closest_indices, closest_points


def extend_bolus_to_initial_area(closest_points, cl, cr, rt, current_area, initial_area, dx, closest_top_indices, cl_outer, cr_outer):
    left_side = closest_points[0]
    right_side = closest_points[1]

    left_wall = np.array([left_side])
    right_wall = np.array([right_side])

    if current_area < initial_area:

        while current_area < initial_area:

            left_wall[:, 1] -= dx
            right_wall[:, 1] -= dx

            cl[:, 1] -= dx
            cr[:, 1] -= dx

            cl_outer[:, 1] -= dx
            cr_outer[:, 1] -= dx

            left_wall = np.row_stack((left_side, left_wall))
            right_wall = np.row_stack((right_wall, right_side))
            bottom_segment = np.row_stack((left_wall, cl, cr, right_wall))
            total_area = np.row_stack((rt[0:closest_top_indices[1]], bottom_segment, rt[closest_top_indices[1]:]))
            current_area = opencv_contour_area(total_area)

    elif current_area >= initial_area:
        bottom_segment = np.row_stack((cl, cr))
        total_area = np.row_stack((rt[0:closest_top_indices[1]], bottom_segment, rt[closest_top_indices[1]:]))
        current_area = opencv_contour_area(total_area)

    # Generating left and right boundaries
    leftlayer1 = np.copy(left_wall)
    leftlayer1[:, 0] -= dx

    lowestpoint = np.copy(leftlayer1[-1, :])
    highestpoint = np.copy(leftlayer1[-1, :])

    for i in range(4):
        # lowestpoint[1] -= dx
        if i <= 4:
            highestpoint[1] += dx
        leftlayer1 = np.row_stack((leftlayer1, lowestpoint, highestpoint))

    leftlayer2 = np.copy(left_wall)
    leftlayer2[:, 0] -= 2 * dx

    lowestpoint = np.copy(leftlayer2[-1, :])
    highestpoint = np.copy(leftlayer2[-1, :])

    for i in range(4):
        # lowestpoint[1] -= dx
        if i <= 4:
            highestpoint[1] += dx
        leftlayer2 = np.row_stack((leftlayer2, lowestpoint, highestpoint))

    leftlayer3 = np.copy(left_wall)
    leftlayer3[:, 0] -= 3 * dx

    lowestpoint = np.copy(leftlayer3[-1, :])
    highestpoint = np.copy(leftlayer2[-1, :])

    for i in range(4):
        # lowestpoint[1] -= dx
        if i <= 4:
            highestpoint[1] += dx
        leftlayer3 = np.row_stack((leftlayer3, lowestpoint, highestpoint))

    rightlayer1 = np.copy(right_wall)
    rightlayer1[:, 0] += dx

    lowestpoint = np.copy(rightlayer1[0, :])
    highestpoint = np.copy(rightlayer1[-1, :])

    for i in range(2):
        # lowestpoint[1] -= dx
        if i <= 2:
            highestpoint[1] += dx
        rightlayer1 = np.row_stack((rightlayer1, lowestpoint))

    rightlayer2 = np.copy(right_wall)
    rightlayer2[:, 0] += 2 * dx

    lowestpoint = np.copy(rightlayer2[0, :])
    highestpoint = np.copy(rightlayer2[-1, :])

    for i in range(2):
        # lowestpoint[1] -= dx
        if i <= 2:
            highestpoint[1] += dx
        rightlayer2 = np.row_stack((rightlayer2, lowestpoint))

    rightlayer3 = np.copy(right_wall)
    rightlayer3[:, 0] += 3 * dx

    lowestpoint = np.copy(rightlayer3[0, :])
    highestpoint = np.copy(rightlayer3[-1, :])

    for i in range(2):
        # lowestpoint[1] -= dx
        if i <= 2:
            highestpoint[1] += dx
        rightlayer3 = np.row_stack((rightlayer3, lowestpoint))

    outer_layers = np.row_stack((leftlayer1, rightlayer1, leftlayer2, rightlayer2, leftlayer3, rightlayer3))
    print(initial_area, current_area)

    return rt, left_wall, cl, cr, right_wall, outer_layers, cl_outer, cr_outer


def interpolate_points(points, num_points):
    tck, _ = splprep(points.T, s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new))


def create_layers(points, num_layers, distance, resolution, num_points):
    layers = [points]
    poly = Polygon(points)
    interpolated_points = interpolate_points(points, num_points)
    a = 2
    for i in range(1, num_layers):
        buffer_poly = poly.buffer(distance * i, cap_style=2, join_style=2, resolution=resolution)

        # Check if the result is a MultiPolygon object
        if isinstance(buffer_poly, MultiPolygon):
            # Choose the largest polygon from the MultiPolygon object
            buffer_poly = max(buffer_poly.geoms, key=lambda p: p.area)

        layer_points = np.array(buffer_poly.exterior.coords)
        layer_points = interpolate_points(layer_points, num_points + a)
        layers.append(layer_points)
        a += 2

    return layers


def create_grid_points(shape_outline, spacing):
    x_min, y_min = np.min(shape_outline, axis=0)
    x_max, y_max = np.max(shape_outline, axis=0)

    x_values = np.arange(x_min, x_max + spacing, spacing)
    y_values = np.arange(y_min, y_max + spacing, spacing)

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    return grid_points


def filter_points_inside_shape(points, shape_outline):
    inside_points = []
    for point in points:
        if cv2.pointPolygonTest(shape_outline, tuple(point), measureDist=False) >= 0:
            inside_points.append(point)
    return np.array(inside_points)


def remove_points_too_close(points, boundary_points, distance_threshold):
    filtered_points = []
    for point in points:
        min_distance = np.min(np.sqrt(np.sum((boundary_points - point) ** 2, axis=1)))
        if min_distance >= distance_threshold:
            filtered_points.append(point)
    return np.array(filtered_points)


def find_closest_point(point, points_array):
    distances = np.linalg.norm(points_array - point, axis=1)
    index = np.argmin(distances)
    return points_array[index], index


def calculate_velocities_single_step(prev_positions, curr_positions, dt):
    velocities = []
    for curr_point in curr_positions:
        closest_prev_point, _ = find_closest_point(curr_point, prev_positions)
        velocity = (curr_point - closest_prev_point) / dt
        velocities.append(velocity)
    return np.array(velocities)


prev_rotated_final = np.array([])
prev_rotated_boundary = np.array([])
initial_bolus_area = 0.0
pos_ref = np.zeros((length, 3))
ShiftEvery = 10
PlotEvery = 1
for i in tqdm(range(length)):
    if i < 0:
        continue
    else:
        image_path = directory + str(i) + '.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Select the largest contour, assuming there's only one shape in the image
        contour = max(contours, key=cv2.contourArea)
        contour_length = cv2.arcLength(contour, closed=True)
        dx = 1.2
        num_points = int(np.ceil(contour_length / dx))

        equally_spaced_points = resample_contour(contour, num_points)
        equally_spaced_points = np.vstack((equally_spaced_points, equally_spaced_points[0]))
        equally_spaced_points[:, 1] = -equally_spaced_points[:, 1]
        contour = contour[:, 0, :]  # Remove extra dimension
        contour[:, 1] = -contour[:, 1]
        # contour = contour.reshape(-1, 1, 2)

        # Determine the bounds of the shape
        x_min, y_min = np.min(equally_spaced_points, axis=0)
        x_max, y_max = np.max(equally_spaced_points, axis=0)

        # Create grid points within the bounds
        spacing = 1.5
        grid_points = create_grid_points(equally_spaced_points, spacing)
        inside_points = filter_points_inside_shape(grid_points, contour)
        filtered_inside_points = remove_points_too_close(inside_points, equally_spaced_points, 1.5)

        # Rotating
        centroid = find_centroid(equally_spaced_points)
        rotated_bolus = rotate(equally_spaced_points, 32, centroid)
        rotated_inside = rotate(filtered_inside_points, 32, centroid)

        # Generate outer layers
        num_outer_layers = 4
        layer_distance = 1.2
        resolution = 16
        outer_layers = np.array(create_layers(rotated_bolus, num_outer_layers, layer_distance, resolution, num_points))
        outer_layers0 = outer_layers[0]
        outer_layers1 = outer_layers[1]
        outer_layers2 = outer_layers[2]
        outer_layers3 = outer_layers[3]

        boundary_matrix = np.row_stack((outer_layers1, outer_layers2, outer_layers3))
        extended_bolus = np.copy(outer_layers0)

        # Extending bolus
        ycut = -480

        if i < 2135:
            x_value = 300
        elif i >= 2135:
            x_value = 340

        rt, cl, cr = cut_bottom_part(outer_layers0, ycut, x_value)
        rt_outer, cl_outer, cr_outer = cut_bottom_part(boundary_matrix, ycut, x_value)

        if i % ShiftEvery == 0:

            # Get 6 of the closest points to the ycut value
            closest_indices_cl, closest_points_cl = find_closest_points(cl, ycut)  # Left cut section
            closest_indices_cr, closest_points_cr = find_closest_points(cr, ycut)  # Right cut section

            closest_y_cl_idx = np.argmax(closest_points_cl[:, 1])
            closest_y_cl = closest_points_cl[closest_y_cl_idx]

            closest_y_cr_idx = np.argmax(closest_points_cr[:, 1])
            closest_y_cr = closest_points_cr[closest_y_cr_idx]

            closest_points = np.array([closest_y_cl, closest_y_cr])

        closest_indices_top, closest_top = find_closest_points(rt, ycut)

        # Create a mask
        mask = closest_top[:, 0] >= x_value

        # Separate the points
        closest_top_cr = closest_top[mask]
        closest_top_cl = closest_top[~mask]

        # Separate the indices
        closest_top_cr_idx = closest_indices_top[mask]
        closest_top_cl_idx = closest_indices_top[~mask]

        # Sort the points and indices based on the y coordinates: top_cl
        sorted_indices_top_cl = np.argsort(closest_top_cl[:, 1])
        sorted_top_cl = closest_top_cl[sorted_indices_top_cl]
        sorted_idx_top_cl = closest_top_cl_idx[sorted_indices_top_cl]

        # Sort the points and indices based on the y coordinates: top_cr
        sorted_indices_top_cr = np.argsort(closest_top_cr[:, 1])
        sorted_top_cr = closest_top_cr[sorted_indices_top_cr]
        sorted_idx_top_cr = closest_top_cr_idx[sorted_indices_top_cr]

        closest_top_indices = np.array([sorted_idx_top_cl[0], sorted_idx_top_cr[0]])

        if i == 800:
            initial_bolus_area = opencv_contour_area(equally_spaced_points)
        current_area = opencv_contour_area(equally_spaced_points)

        if current_area < initial_bolus_area:
            rt, left_wall, cl, cr, right_wall, outer_layers_wall, cl_outer, cr_outer = extend_bolus_to_initial_area(closest_points, cl, cr, rt, current_area, initial_bolus_area, dx, closest_top_indices, cl_outer, cr_outer)

            extended_bolus = np.row_stack((rt, left_wall, cl, cr, right_wall))
            outer_rt = create_layers(rt, 4, layer_distance, resolution, num_points)
            outer_rt0 = outer_rt[0]
            outer_rt1 = outer_rt[1]
            outer_rt2 = outer_rt[2]
            outer_rt3 = outer_rt[3]
            outer_rt_matrix = np.row_stack((outer_rt1, outer_rt2, outer_rt3))

            outer_cl = create_layers(cl, 4, layer_distance, resolution, num_points)
            outer_cl0 = outer_cl[0]
            outer_cl1 = outer_cl[1]
            outer_cl2 = outer_cl[2]
            outer_cl3 = outer_cl[3]
            outer_cl_matrix = np.row_stack((outer_cl1, outer_cl2, outer_cl3))

            outer_cr = create_layers(cr, 4, layer_distance, resolution, num_points)
            outer_cr0 = outer_cr[0]
            outer_cr1 = outer_cr[1]
            outer_cr2 = outer_cr[2]
            outer_cr3 = outer_cr[3]
            outer_cr_matrix = np.row_stack((outer_cr1, outer_cr2, outer_cr3))

            boundary_matrix = np.row_stack((rt_outer, cl_outer, cr_outer, outer_layers_wall))

        # Rotate Shape Back
        rotated_final = rotate(extended_bolus, -32, centroid)
        rotated_boundary = rotate(boundary_matrix, -32, centroid)

        # Calculate the velocities if there are previous positions available
        if prev_rotated_final.size > 0 and prev_rotated_boundary.size > 0:
            final_velocities = calculate_velocities_single_step(prev_rotated_final, rotated_final, dt)
            boundary_velocities = calculate_velocities_single_step(prev_rotated_boundary, rotated_boundary, dt)
            # np.save(directory3 + str(i - 800) + '.npy', final_velocities)
            # np.save(directory4 + str(i - 800) + '.npy', boundary_velocities)
        else:
            final_velocities = np.zeros_like(rotated_final)
            boundary_velocities = np.zeros_like(rotated_boundary)
            # np.save(directory3 + str(i - 800) + '.npy', final_velocities)
            # np.save(directory4 + str(i - 800) + '.npy', boundary_velocities)

        # Update the previous positions with the current positions
        prev_rotated_final = rotated_final.copy()
        prev_rotated_boundary = rotated_boundary.copy()

        # np.save(directory1 + str(i-800) + '.npy', rotated_final)
        # np.save(directory2 + str(i-800) + '.npy', rotated_boundary)

        if i % PlotEvery == 0:
            plt.scatter(rotated_final[:, 0], rotated_final[:, 1], s=1, c='red', label='Initial')
            plt.scatter(rotated_boundary[:, 0], rotated_boundary[:, 1], s=1, c='lightcoral', label='Ghost Points')
            # plt.scatter(rotated_initial[:, 0], rotated_initial[:, 1], s=1)
            # plt.scatter(equally_spaced_points[:, 0], equally_spaced_points[:, 1], s=1, c='lightcoral', label='Initial')
            plt.xlim(175, 500)
            plt.ylim(-250, -680)
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.show()
            plt.tight_layout()
            plt.pause(0.0001)
            plt.draw()
            plt.clf()


