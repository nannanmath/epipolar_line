import numpy as np
import cv2

def left_hand_to_right_hand(coordinates):
    """
    将左手系坐标转换为右手系坐标。
    
    参数:
    coordinates (tuple): 左手系坐 (x,y,z)。
    
    返回:
    tuple: 右手系坐标，形状为 (x,y,z)。
    """
    # 假设左手系和右手系的区别在于 X 轴方向相反
    # 因此，将 X 轴坐标取反
    x, y, z = coordinates
    x = -x
    y = y
    z = z
    
    return (x, y, z)


def left_hand_to_right_hand_euler(euler_angles):
    """
    将左手系欧拉角转换为右手系欧拉角。
    
    参数:
    euler_angles (tuple): 左手系欧拉角 (pitch, yaw, roll)。
    
    返回:
    tuple: 右手系欧拉角 (pitch, yaw, roll)。
    """
    pitch, yaw, roll = euler_angles
    
    # 假设左手系和右手系的区别在于 yaw 和 roll 方向相反
    # 因此，将 yaw 和 roll 取反
    right_hand_pitch = pitch
    right_hand_yaw = -yaw
    right_hand_roll = -roll + 180
    
    return (right_hand_pitch, right_hand_yaw, right_hand_roll)

def euler_to_rotation_matrix(euler_angles):
    """
    将欧拉角转换为旋转矩阵。
    
    参数:
    euler_angles (tuple): 欧拉角 (pitch, yaw, roll)，单位为度。
    
    返回:
    numpy.ndarray: 旋转矩阵，形状为 (3, 3)。
    """
    pitch, yaw, roll = np.radians(euler_angles)  # 将角度转换为弧度
    
    # 计算绕X轴（滚转）的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # 计算绕Y轴（俯仰）的旋转矩阵
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # 计算绕Z轴（偏航）的旋转矩阵
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = np.dot(Ry, np.dot(Rx, Rz))
    
    return R

def skew_symmetric(t):
    """
    根据向量 t 构建反对称矩阵（叉乘矩阵）
    """
    return np.array([
        [0, -t[2], t[1]],
        [t[2],  0, -t[0]],
        [-t[1], t[0],  0]
    ])

def compute_essential_matrix(R1, t1, R2, t2):
    """
    计算本质矩阵 E
    """
    # 计算相对旋转矩阵和位移向量
    R_rel = np.dot(R2, R1.T)
    t_rel = t2 - np.dot(R_rel, t1)
    # 构建反对称矩阵
    t_skew = skew_symmetric(t_rel)
    # 计算本质矩阵
    E = np.dot(t_skew, R_rel)
    return E

def compute_fundamental_matrix_core(E, K1, K2):
    """
    计算基础矩阵 F
    """
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T
    F = np.dot(np.dot(K2_inv_T, E), K1_inv)
    return F

def compute_fundamental_matrix(K1, R1, t1, K2, R2, t2):
    """
    计算基础矩阵。
    
    参数:
    K1 (numpy.ndarray): 第一个相机的内参矩阵，形状为 (3, 3)。
    R1 (numpy.ndarray): 第一个相机的外参旋转矩阵，形状为 (3, 3)。
    t1 (numpy.ndarray): 第一个相机的外参平移向量，形状为 (3, 1)。
    K2 (numpy.ndarray): 第二个相机的内参矩阵，形状为 (3, 3)。
    R2 (numpy.ndarray): 第二个相机的外参旋转矩阵，形状为 (3, 3)。
    t2 (numpy.ndarray): 第二个相机的外参平移向量，形状为 (3, 1)。
    
    返回:
    numpy.ndarray: 基础矩阵，形状为 (3, 3)。
    """
    # 计算本质矩阵
    E = compute_essential_matrix(R1, t1, R2, t2)
    
    # 计算基础矩阵
    F = compute_fundamental_matrix_core(E, K1, K2)
    
    return F

def compute_epipolar_constraint(points1, points2, F):
    """
    计算对极约束。
    
    参数:
    points1 (numpy.ndarray): 第一幅图像中的点，形状为 (N, 2)，其中 N 是点的数量。
    points2 (numpy.ndarray): 第二幅图像中的点，形状为 (N, 2)，其中 N 是点的数量。
    F (numpy.ndarray): 基础矩阵，形状为 (3, 3)。
    
    返回:
    numpy.ndarray: 对极约束的误差，形状为 (N,)。
    """
    # 将点转换为齐次坐标
    points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_homogeneous = np.hstack([points2, np.ones((points2.shape[0], 1))])
    
    # 计算对极约束误差
    errors = np.sum(points2_homogeneous * np.dot(F, points1_homogeneous.T).T, axis=1)
    
    return errors

def compute_epilines(points, F):
    """
    计算极线。
    
    参数:
    points (numpy.ndarray): 图像中的点，形状为 (N, 2)，其中 N 是点的数量。
    F (numpy.ndarray): 基础矩阵，形状为 (3, 3)。
    
    返回:
    list: 极线系数，每个元素是一个包含极线参数的元组 (a, b, c)。
    """
    # 将点转换为齐次坐标
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 计算极线
    epilines = np.dot(F, points_homogeneous.T).T
    
    # 归一化极线参数
    epilines = epilines / np.linalg.norm(epilines[:, :2], axis=1)[:, np.newaxis]
    
    return epilines

def draw_epilines(img, epilines, color=(0, 255, 0)):
    """
    在图像中绘制极线。
    
    参数:
    img (numpy.ndarray): 输入图像。
    epilines (list): 极线系数，每个元素是一个包含极线参数的元组 (a, b, c)。
    color (tuple): 极线的颜色，默认为绿色 (0, 255, 0)。
    
    返回:
    numpy.ndarray: 绘制了极线的图像。
    """
    height, width = img.shape[:2]
    
    for line in epilines:
        a, b, c = line
        # 计算极线与图像边界的交点
        x0, y0 = map(int, [0, -c / b])
        x1, y1 = map(int, [width, -(c + a * width) / b])
        
        # 绘制极线
        cv2.line(img, (x0, y0), (x1, y1), color, 2)
    
    return img

def compute_intrinsic_matrix(fov, image_width, image_height):
    """
    根据视场角和图像分辨率计算相机内参矩阵。
    
    参数:
    fov (float): 垂直视场角（度）。
    image_width (int): 图像宽度（像素）。
    image_height (int): 图像高度（像素）。
    
    返回:
    numpy.ndarray: 相机内参矩阵，形状为 (3, 3)。
    """
    # 将视场角转换为弧度
    fov_rad = np.radians(fov)
    
    # 计算焦距
    f = image_height / (2 * np.tan(fov_rad / 2))
    
    # 计算内参矩阵
    K = np.array([
        [f, 0, image_width / 2],
        [0, f, image_height / 2],
        [0, 0, 1]
    ])
    
    return K

def draw_point_icon(img, point, radius=5, color=(0, 0, 255), thickness=-1):
    """
    在图像的指定像素点处绘制一个点图标。
    
    参数:
    img (numpy.ndarray): 输入图像。
    point (tuple): 指定像素点的坐标 (x, y)。
    radius (int): 点的半径，默认为 5。
    color (tuple): 点的颜色，默认为红色 (0, 0, 255)。
    thickness (int): 点的厚度，默认为 -1（填充）。
    
    返回:
    numpy.ndarray: 绘制了点图标的图像。
    """
    x, y = point
    
    # 绘制点图标
    cv2.circle(img, (x, y), radius, color, thickness)
    
    return img

if __name__ == "__main__":
    fov = 60
    image_width = 2545
    image_height = 1080
    # cam1:
    image1_path = "data/cam1_steel_0005.jpg" # 2545x1080
    pitch1 = 20.423
    yaw1 = 0
    roll1 = 0
    x1 = -83.3
    y1 = 11.6
    z1 = 52.8
 
    # cam2:
    image2_path = "data/cam2_steel_0005.jpg"
    pitch2 = 33.198
    yaw2 = 180
    roll2 = 0
    x2 = -105.7
    y2 = 16.29
    z2 = 143.4


    # 相机内参：
    K = compute_intrinsic_matrix(fov, image_width, image_height)

    # 将cam1和cam2的位姿由左手系转右手系
    # 位置:
    left_hand_coordinates1 = (x1, y1, z1)
    right_hand_coordinates1 = left_hand_to_right_hand(left_hand_coordinates1)
    left_hand_coordinates2 = (x2, y2, z2)
    right_hand_coordinates2 = left_hand_to_right_hand(left_hand_coordinates2)


    # 欧拉角:
    left_hand_euler1 = (pitch1, yaw1, roll1)
    right_hand_euler1 = left_hand_to_right_hand_euler(left_hand_euler1)
    left_hand_euler2 = (pitch2, yaw2, roll2)
    right_hand_euler2 = left_hand_to_right_hand_euler(left_hand_euler2)

    # 由欧拉角计算旋转矩阵
    R1 = euler_to_rotation_matrix(right_hand_euler1)
    R1 = np.linalg.inv(R1)
    R2 = euler_to_rotation_matrix(right_hand_euler2)
    R2 = np.linalg.inv(R2)
    t1 = np.array(right_hand_coordinates1)
    t1 = -np.dot(R1, t1)
    t2 = np.array(right_hand_coordinates2)
    t2 = -np.dot(R2, t2)

    P_w_left = (-78.92, 1.04, 80)
    P_w_right = np.array(left_hand_to_right_hand(P_w_left))

    P_cam1 = np.dot(R1, P_w_right) + t1
    P_image1 = np.dot(K, P_cam1) # 齐次
    P_image1 = [int(x / P_image1[2]) for x in P_image1[:2]]
    image1 = cv2.imread(image1_path)
    image1_with_point = draw_point_icon(image1, P_image1)
    cv2.imwrite("image1_with_point.jpg", image1_with_point)

    P_cam2 = np.dot(R2, P_w_right) + t2
    P_image2 = np.dot(K, P_cam2) # 齐次
    P_image2 = [int(x / P_image2[2]) for x in P_image2[:2]]
    image2 = cv2.imread(image2_path)
    image2_with_point = draw_point_icon(image2, P_image2)
    cv2.imwrite("image2_with_point.jpg", image2_with_point)


    # exit(0)



    # 计算
    # 计算基础矩阵
    F = compute_fundamental_matrix(K, R1, t1, K, R2, t2)
    # 计算对极约束误差
    P_image1 = np.array([P_image1])
    P_image2 = np.array([P_image2])
    errors = compute_epipolar_constraint(P_image1, P_image2, F)
    # 在image2中绘制极线
    # 求image2中极线系数
    epilines = compute_epilines(P_image1, F)
    # 绘制极线
    image2 = cv2.imread(image2_path)
    img = draw_epilines(image2, epilines)
    cv2.imwrite("epi_image.jpg", img)


   
