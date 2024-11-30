




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])





# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the aspect ratio of the plot box to be equal
ax.set_box_aspect([1, 1, 1])







# If df_dx_dy is actually zero, that might be the reason for this coincidendence


# Define the function
def f(x, y):
    return x**2 + y**3

# Calculate partial derivatives
def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 3 * y**2

def df_dx_dx(x, y):
    return 2

def df_dy_dy(x, y):
    return 6 * y

def df_dx_dy(x, y):
    return 0




# # Define the function
# def f(x, y):
#     return x**5 + y**3

# # Calculate partial derivatives
# def df_dx(x, y):
#     return 5 * x**4

# def df_dy(x, y):
#     return 3 * y**2

# def df_dx_dx(x, y):
#     return 20 * x**3

# def df_dy_dy(x, y):
#     return 6 * y

# def df_dx_dy(x, y):
#     return 0





# # Define the function
# def f(x, y):
#     return 0.01 * x**2 + y**3

# # Calculate partial derivatives
# def df_dx(x, y):
#     return 0.02 * x

# def df_dy(x, y):
#     return 3 * y**2

# def df_dx_dx(x, y):
#     return 0.02

# def df_dy_dy(x, y):
#     return 6 * y

# def df_dx_dy(x, y):
#     return 0






# # Define the function
# def f(x, y):
#     return 0.01 * x**2 + y**3 + 5 * x * y

# # Calculate partial derivatives
# def df_dx(x, y):
#     return 0.02 * x + 5 * y

# def df_dy(x, y):
#     return 3 * y**2 + 5 * x

# def df_dx_dx(x, y):
#     return 0.02

# def df_dy_dy(x, y):
#     return 6 * y

# def df_dx_dy(x, y):
#     return 5






# # Define the function
# def f(x, y):
#     return x**2 * y**3

# # Calculate partial derivatives
# def df_dx(x, y):
#     return 2 * x * y**3

# def df_dy(x, y):
#     return 3 * x**2 * y**2

# def df_dx_dx(x, y):
#     return 2 * y**3

# def df_dy_dy(x, y):
#     return 6 * x**2 * y

# def df_dx_dy(x, y):
#     return 6 * x * y**2











# Point of tangency
# x0, y0 = 2, 2
point=0.8
x0, y0 = point, point
z0 = f(x0, y0)
start = np.array([x0, y0, z0])

# Tangent plane equation
def tangent_plane(x, y):
    return z0 + df_dx(x0, y0) * (x - x0) + df_dy(x0, y0) * (y - y0)

# Tangent paraboloid equation
def tangent_paraboloid(x, y):
    return z0 + df_dx(x0, y0) * (x - x0) + df_dy(y0, y0) * (y - y0) + 0.5 * (df_dx_dx(x0, y0) * (x - x0)**2 + df_dy_dy(x0, y0) * (y - y0)**2 + 2 * df_dx_dy(x0, y0) * (x - x0) * (y - y0))

# Create a grid for plotting
len = 0.9
size = (point-len, point+len)

x = np.linspace(size[0], size[1], 100)
y = np.linspace(size[0], size[1]+0.1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Calculate the tangent plane
Z_tangent = tangent_plane(X, Y)

# Calculate the tangent paraboloid
Z_paraboloid = tangent_paraboloid(X, Y)

gradient = np.array([df_dx(x0, y0), df_dy(x0, y0)])
hessian = np.array([[df_dx_dx(x0, y0), df_dx_dy(x0, y0)], [df_dx_dy(x0, y0), df_dy_dy(x0, y0)]])
parabola_min = np.array([point, point]) - np.linalg.solve(hessian, gradient)

print(f'Gradient: {gradient}')
print(f'Hessian: {hessian}')
print(f'Parabola minimum: {parabola_min}')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, rstride=100, cstride=100)

# Plot the tangent plane
ax.plot_surface(X, Y, Z_tangent, color='r', alpha=0.5)

# Plot the tangent paraboloid
ax.plot_surface(X, Y, Z_paraboloid, color='g', alpha=0.5)

# Highlight the point of tangency
ax.scatter(x0, y0, z0, color='k', s=50)

# Highlight (0,0,0)
ax.scatter(0,0,0, color='r', s=50)

# Highlight the minimum of the paraboloid
ax.scatter(parabola_min[0], parabola_min[1], tangent_paraboloid(*parabola_min), color='g', s=50)









# Define the plane using vectors (1, 1, 2) and (0, 0, 1)
u = start
v = np.array([0, 0, 1])

plane_X = X * u[0] + Y * v[0]
plane_Y = X * u[1] + Y * v[1]
plane_Z = X * u[2] + Y * v[2]


# Plot the cutting plane
ax.plot_surface(plane_X, plane_Y, plane_Z, color='b', alpha=0.5, label='Plane')

print(f'u: {u}')
print(f'v: {v}')





# plane_gradient = np.array([1, 1, df_dx(x0, y0) + df_dy(x0, y0)]) # this is wrong. it isnt the perpendicular direction . it is literally up the plane.
plane_gradient = np.array([df_dx(x0, y0), df_dy(x0, y0), 0])
z_unit = np.array([0, 0, 1])


print(f'Plane gradient: {plane_gradient}')
print(f'Z unit: {z_unit}')
print(f'Start: {start}')

# plane_X = X * (plane_gradient[0] - start[0]) + Y * (z_unit[0]- start[1]) + start[2]
# plane_Y = X * (plane_gradient[1] - start[0]) + Y * (z_unit[1]- start[1]) + start[2]
# plane_Z = X * (plane_gradient[2] - start[0]) + Y * (z_unit[2] - start[1]) + start[2]

# plane_X = (X - start[0]) * plane_gradient[0] + (Y - start[1]) * z_unit[0] - start[2]
# plane_Y = (X - start[0]) * plane_gradient[1] + (Y - start[1]) * z_unit[1] - start[2]
# plane_Z = (X - start[0]) * plane_gradient[2] + (Y - start[1]) * z_unit[2] - start[2]

x_move = -1
y_move = -1
new_x = X + x_move
new_y = Y + y_move
plane_X = new_x * plane_gradient[0] + new_y * z_unit[0] + start[0]
plane_Y = new_x * plane_gradient[1] + new_y * z_unit[1] + start[1]
plane_Z = new_x * plane_gradient[2] + new_y * z_unit[2] + start[2]

# plane_X = X * plane_gradient[0] + Y * z_unit[0]
# plane_Y = X * plane_gradient[1] + Y * z_unit[1]
# plane_Z = X * plane_gradient[2] + Y * z_unit[2] 


ax.plot_surface(plane_X, plane_Y, plane_Z, color='r', alpha=0.5, label='Plane')



# This is wrong - just because you can linearly extrapolate to any x,y point, that doesn't mean it is on the gradient direction.

# plane_grad_point = start - 0.05 * np.array([1, 1, tangent_plane(start[0] + 1, start[1]+1)])     # np.array([1, 1, df_dx(x0, y0) + df_dy(x0, y0)])
# ax.scatter(plane_grad_point[0], plane_grad_point[1], plane_grad_point[2], color='g', s=50)

# print(f'Plane grad point: {plane_grad_point}')



plane_grad_point = start - 0.05 * np.array([plane_gradient[0], plane_gradient[1], tangent_plane(start[0] + plane_gradient[0], start[1]+plane_gradient[1])])     # np.array([1, 1, df_dx(x0, y0) + df_dy(x0, y0)])
ax.scatter(plane_grad_point[0], plane_grad_point[1], plane_grad_point[2], color='g', s=50)

print(f'Plane grad point: {plane_grad_point}')













# Set plot titles and labels
ax.set_title('Surface and Tangent Plane at (2, 2)')
# ax.set_zlim(0, 30)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


set_axes_equal(ax)

# Display the plot
plt.show()