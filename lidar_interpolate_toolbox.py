import numpy as np
import scipy
import cv2
from sklearn.neighbors import KNeighborsRegressor
import scipy.interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def geo_interp(lidar_image, theta_spatial=0.5, grid=5, epochs=4):
    assert (grid > 1)
    lidar_image = np.pad(lidar_image, ((grid, grid), (grid, grid)), 'edge')
    geo_dis_map = np.full((lidar_image.shape[0], lidar_image.shape[1], 2), fill_value=-1, dtype=np.float)
    geo_dis_map[:, :, 1] = lidar_image[:, :]

    for i in range(0, lidar_image.shape[0]):
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                geo_dis_map[i][j][0] = 0.

    bias = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for epoch in range(epochs):
        # forward
        for i in range(1, lidar_image.shape[0] - 1):
            for j in range(1, lidar_image.shape[1] - 1):
                if lidar_image[i][j] > 0:
                    continue
                min_geo_dis = 1000000.
                loc_val = -1.
                for (x, y) in bias[:4]:
                    if geo_dis_map[i + x][j + y][0] != -1.:
                        spatial_dis = np.sqrt(x ** 2 + y ** 2)
                        geo_dis = theta_spatial * spatial_dis + geo_dis_map[i + x][j + y][0]
                        if geo_dis < min_geo_dis:
                            loc_val = geo_dis
                            geo_dis_map[i][j][1] = geo_dis_map[i + x][j + y][1]
                geo_dis_map[i][j][0] = loc_val
        # backward
        for i in range(lidar_image.shape[0] - 2, 0, -1):
            for j in range(lidar_image.shape[0] - 2, 0, -1):
                if lidar_image[i][j] > 0:
                    continue
                min_geo_dis = 1000000.
                loc_val = -1.
                for (x, y) in bias[4:8]:
                    if geo_dis_map[i + x][j + y][0] != -1.:
                        spatial_dis = np.sqrt(x ** 2 + y ** 2)
                        geo_dis = theta_spatial * spatial_dis + geo_dis_map[i + x][j + y][0]
                        if geo_dis < min_geo_dis:
                            loc_val = geo_dis
                            geo_dis_map[i][j][1] = geo_dis_map[i + x][j + y][1]
                geo_dis_map[i][j][0] = loc_val

    return geo_dis_map[grid:lidar_image.shape[0] - grid, grid:lidar_image.shape[1] - grid, 1]


def spatial_without_color_interp(lidar_image, grid=7, sigma_spatial=2.0):
    lidar_smooth_result = np.zeros((lidar_image.shape[0], lidar_image.shape[1]), dtype=np.uint8)
    geo_map = geo_interp(lidar_image)
    lidar_image = np.pad(lidar_image, ((grid, grid), (grid, grid)), 'edge')
    geo_map = np.pad(geo_map, ((grid, grid), (grid, grid)), 'edge')
    for i in range(grid, lidar_image.shape[0] - grid):
        for j in range(grid, lidar_image.shape[1] - grid):
            depth_slice = lidar_image[i - grid:i + grid, j - grid:j + grid]
            geo_slice = geo_map[i - grid:i + grid, j - grid:j + grid]
            if np.abs(geo_slice.max() - geo_slice.min()) < 1e-4:
                sigma_depth = 1
            else:
                sigma_depth = - np.log(geo_slice.max() - geo_slice.min())
            dep_weight = np.exp(-(depth_slice - geo_map[i][j]) ** 2 / (2 * sigma_depth ** 2))
            spa_weight = np.zeros_like(dep_weight)
            for k in range(-int(grid), int(grid + 1)):
                for l in range(-int(grid), int(grid + 1)):
                    spa_weight[k][l] = np.exp(-(k ** 2 + l ** 2) / (2 * sigma_spatial ** 2))
            lidar_smooth_result[i - grid, j - grid] = np.sum(depth_slice * dep_weight * spa_weight)
    return lidar_smooth_result


def interpolator2d(lidar_image, kind='linear'):
    m, n = lidar_image.shape
    points = list()
    for i in range(0, lidar_image.shape[0]):
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                points.extend([[i, j, lidar_image[i][j]]])
    if len(points) < 4:
        return np.zeros((lidar_image.shape[0], lidar_image.shape[1]), dtype=np.uint8)
    points = np.asarray(points)
    ij, d = points[:, :-1], points[:, 2]
    if kind == 'linear':
        f = scipy.interpolate.LinearNDInterpolator(ij, d, fill_value=0)
    elif kind == 'nearest':
        f = scipy.interpolate.NearestNDInterpolator(ij, d)
    elif kind == 'clough':
        f = scipy.interpolate.CloughTocher2DInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(lidar_image.shape)
    return disparity


def barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def barycentric_interp(lidar_image):
    rect = (0, 0, lidar_image.shape[1], lidar_image.shape[0])
    lidar_smooth_result = np.zeros((lidar_image.shape[0], lidar_image.shape[1]), dtype=np.uint8)
    subdiv = cv2.Subdiv2D(rect)
    for i in range(0, lidar_image.shape[0]):
        index = i * lidar_image.shape[1]
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                subdiv.insert([(j, i)])
            index += 1
    for i in range(0, lidar_image.shape[0]):
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                lidar_smooth_result[i][j] = lidar_image[i][j]
            else:
                loc, edge, vertex = subdiv.locate((j, i))
                neighbor_point = []
                for k in range(0, 3):
                    neighbor_point.extend([np.array(subdiv.getVertex(subdiv.edgeOrg(edge)[0])[0], dtype=np.int)])
                    edge = subdiv.getEdge(edge, cv2.Subdiv2D_NEXT_AROUND_LEFT)

                valid_points = np.full(3, fill_value=False, dtype=np.bool)

                for k in range(0, 3):
                    if 0 <= neighbor_point[k][0] < lidar_image.shape[1] and 0 <= neighbor_point[k][1] < \
                            lidar_image.shape[0]:
                        valid_points[k] = True

                if np.all(valid_points):
                    u, v, w = barycentric(np.array([j, i]), neighbor_point[0], neighbor_point[1], neighbor_point[2])
                    lidar_smooth_result[i][j] = lidar_image[neighbor_point[0][1]][neighbor_point[0][0]] * u + \
                                                lidar_image[neighbor_point[1][1]][neighbor_point[1][0]] * v + \
                                                lidar_image[neighbor_point[2][1]][neighbor_point[2][0]] * w
    return lidar_smooth_result


def knn_interp(lidar_image, k=1, p=2.):
    points = list()
    values = list()
    lidar_smooth_result = np.zeros((lidar_image.shape[0], lidar_image.shape[1]), dtype=np.uint8)
    knr = KNeighborsRegressor(n_neighbors=k, p=p)
    for i in range(0, lidar_image.shape[0]):
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                points.extend([[i, j]])
                values.extend([lidar_image[i][j]])

    knr.fit(points, values)
    for i in range(0, lidar_image.shape[0]):
        for j in range(0, lidar_image.shape[1]):
            if lidar_image[i][j] > 0:
                lidar_smooth_result[i][j] = lidar_image[i][j]
            else:
                lidar_smooth_result[i][j] = knr.predict([[i, j]])[0]

    return lidar_smooth_result


def grid_weight_interp(lidar_image, grid=7):
    lidar_smooth_result = np.zeros((lidar_image.shape[0], lidar_image.shape[1]), dtype=np.uint8)
    lidar_image = np.pad(lidar_image, ((grid, grid), (grid, grid)), 'edge')
    for i in range(grid, lidar_image.shape[0] - grid):
        for j in range(grid, lidar_image.shape[1] - grid):
            X = 0
            Y = 0
            for k in range(-int(grid), int(grid + 1)):
                for l in range(-int(grid), int(grid + 1)):
                    if k == l == 0:
                        continue
                    s = 1. / np.sqrt(k ** 2 + l ** 2)
                    X += s
                    Y += lidar_image[i + k][j + l] * s
            lidar_smooth_result[i - grid, j - grid] = Y / X
    return lidar_smooth_result


def lattice(m, n, connect):
    if m * n == 1:
        return [(0, 0)], []
    N = m * n

    assert (connect < 2)

    if connect < 2:
        x = range(m)
        y = range(n)
        X, Y = np.meshgrid(x, y)
        points = np.array(list(zip(X.flatten(), Y.flatten())))
        edges = [(i, i + 1) for i in range(1, N + 1)]
        edges.extend([(i, i + m) for i in range(1, N + 1)])
        if connect == 1:
            border = np.linspace(1, N + 1, N)
            border1 = np.where((border % m - 1) > 0)
            border2 = np.where((border % m) > 0)
            border3 = border1 + m - 1
            border4 = border2 + m + 1
            edges.extend(list(zip(border1, border3)))
            edges.extend(list(zip(border2, border4)))
        edges = np.asarray(edges)
        val_ind = np.bitwise_or(np.bitwise_or(np.bitwise_or(edges[:, 0] > N, edges[:, 0] < 1), edges[:, 1] > N),
                                edges[:, 1] < 1)
        for i in range(m, N, m):
            val_ind[i] = True
        val_ind = np.repeat(val_ind.reshape(-1, 1), 2, axis=1)
        marr = np.ma.MaskedArray(edges, mask=val_ind)
        edges = np.ma.compress_rows(marr)

    return points, edges


def make_weight_l2(edges, vals, val_scale, points=None, geo_scale=0, epsilon=1e-5):
    if val_scale > 0:
        val_dis = np.asarray([np.sum((vals[edges[i, 0] - 1] - vals[edges[i, 1] - 1]) ** 2) for i in range(len(edges))])
        if val_dis.max() != val_dis.min():
            val_dis = (val_dis - val_dis.min()) / (val_dis.max() - val_dis.min())
    else:
        val_scale = 0
        val_dis = np.zeros((edges.shape[0]))

    if geo_scale != 0:
        geo_dis = np.asarray(
            [np.sum((points[edges[i, 0] - 1] - points[edges[i, 1] - 1]) ** 2) for i in range(len(edges))])
        if geo_dis.max() != geo_dis.min():
            geo_dis = (geo_dis - geo_dis.min()) / (geo_dis.max() - geo_dis.min())
    else:
        geo_scale = 0
        geo_dis = np.zeros((edges.shape[0]))

    weights = np.exp(-np.add(geo_scale * geo_dis, val_scale * val_dis)) + epsilon
    return weights


def adjacency(edges, weights, N):
    return sparse.coo_matrix((np.append(weights, weights),
                              (np.append(edges[:, 0] - 1, edges[:, 1] - 1),
                               np.append(edges[:, 1] - 1, edges[:, 0] - 1))),
                             shape=(N, N))


def sd_filter(img, lidar_image, connect=0, lamda=10, mu=500, nu=200, step=2, is_sparse=True):
    img = img / 255.
    lidar_image = lidar_image / 255.
    u0 = np.ones_like(lidar_image)
    _, edges = lattice(lidar_image.shape[0], lidar_image.shape[1], connect)

    N = lidar_image.shape[0] * lidar_image.shape[1]

    f_vals = lidar_image.transpose(1, 0).reshape(N)
    if is_sparse:
        A = np.zeros_like(f_vals)
        A[f_vals > 0] = 1
        C = sparse.coo_matrix((A, (range(0, N), range(0, N))))
        F = C @ f_vals.astype(np.float)
    else:
        C = sparse.coo_matrix((np.ones(N), (range(0, N), range(0, N))))
        F = C @ f_vals.astype(np.float)

    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    g_vals = img.transpose(1, 0).reshape(N)
    weights_g = make_weight_l2(edges, g_vals, mu)
    g_w = adjacency(edges, weights_g, N)
    for i in tqdm(range(step), desc='sdfilter step'):
        u_vals = u0.transpose(1, 0).reshape(N)
        weights_u = make_weight_l2(edges, u_vals, nu)
        u_w = adjacency(edges, weights_u, N)
        W = g_w.multiply(u_w)
        D = sparse.coo_matrix((W.sum(axis=0).A1, (range(0, N), range(0, N))))
        L = D - W
        R = C + lamda * L
        U = spsolve(R, F)
        u = U.reshape((lidar_image.shape[1], lidar_image.shape[0])).transpose((1, 0))
        u0 = u
    return u * 255.


def atgv(reconstructed_lidar_image, lidar_image, gray,
         kernel=np.multiply(cv2.getGaussianKernel(5, 1), (cv2.getGaussianKernel(5, 1)).T),
         weight_scale=40., alpha1=20., alpha2=0.2,
         beta=10., gamma=0.85, l=1., step=200, min_val=1e-8, tau=1., eta_p=4., eta_q=2.):
    # self defined anisotropic diffusion
    def dx(u):
        return np.append(u[:, 1:], u[:, 0].reshape((u.shape[0], 1)), axis=1) - u

    def dxz(u):
        return np.append(u[:, :-1], np.zeros_like(u[:, -1].reshape((u.shape[0], 1))), axis=1) - \
               np.append(np.zeros_like(u[:, -1].reshape((u.shape[0], 1))), u[:, :-1], axis=1)

    def dy(u):
        return np.append(u[1:, :], u[0, :].reshape((1, u.shape[1])), axis=0) - u

    def dyz(u):
        return np.append(u[:-1, :], np.zeros_like(u[-1, :].reshape((1, u.shape[1]))), axis=0) - \
               np.append(np.zeros_like(u[-1, :].reshape((1, u.shape[1]))), u[:-1, :], axis=0)

    M, N = lidar_image.shape[0], lidar_image.shape[1]
    sigma = 1. / tau
    weight = np.zeros((lidar_image.shape[0], lidar_image.shape[1]))
    weight[lidar_image > 0] = weight_scale
    gray = gray / 255.
    grad_x = cv2.filter2D(gray, -1, kernel)
    grad_y = cv2.filter2D(gray, -1, kernel.transpose((1, 0)))
    abs_img = np.sqrt(grad_x ** 2 + grad_y ** 2)
    n = np.asarray(list(zip(grad_x.flatten(), grad_y.flatten())))
    norm_n = np.sqrt(np.sum(n, axis=1))
    norm_n = np.asarray(list(zip(norm_n.flatten(), norm_n.flatten())))
    norm_n[norm_n[:, 0] < min_val] = 1
    norm_n[norm_n[:, 1] < min_val] = 0
    norm_n_T = np.asarray(list(zip(norm_n[:, 1], -norm_n[:, 0])))
    W = np.exp(-beta * abs_img ** gamma)
    W[W < min_val] = min_val
    W = W.flatten()
    grad_i = W * norm_n[:, 0] ** 2 + norm_n_T[:, 0] ** 2
    grad_j = W * norm_n[:, 0] * norm_n[:, 1] + norm_n_T[:, 0] * norm_n_T[:, 1]
    grad_k = W * norm_n[:, 1] ** 2 + norm_n_T[:, 1] ** 2
    grad_i = grad_i.reshape((M, N))
    grad_j = grad_j.reshape((M, N))
    grad_k = grad_k.reshape((M, N))

    eta_u = (grad_i ** 2 + grad_j ** 2 + 2 * grad_k ** 2 +
             (grad_i + grad_k) ** 2 + (grad_j + grad_k) ** 2) * (alpha1 ** 2)
    eta_v = np.zeros((lidar_image.shape[0], lidar_image.shape[1], 2))
    eta_v[:, :, 0] = (alpha2 ** 2) * (grad_j ** 2 + grad_k ** 2) + 4 * alpha1 ** 2
    eta_v[:, :, 1] = (alpha2 ** 2) * (grad_i ** 2 + grad_k ** 2) + 4 * alpha1 ** 2
    p = np.zeros((M, N, 2))
    q = np.zeros((M, N, 4))
    u = reconstructed_lidar_image / 255.
    u_ = u
    v = np.zeros((M, N, 2))
    v_ = v

    grad_v = np.zeros((M, N, 4))
    dw = lidar_image / 255. * weight

    for i in tqdm(range(step), desc='tgv step'):
        if sigma < 1000:
            mu = 1 / np.sqrt(1 + 0.7 * tau * l)
        else:
            mu = 1
        u_x = dx(u_) - v_[:, :, 0]
        u_y = dy(u_) - v_[:, :, 1]

        du_tensor_x = grad_i * u_x + grad_k * u_y
        du_tensor_y = grad_k * u_x + grad_j * u_y

        p[:, :, 0] = p[:, :, 0] + alpha2 * sigma / eta_p * du_tensor_x
        p[:, :, 1] = p[:, :, 1] + alpha2 * sigma / eta_p * du_tensor_y

        reprojection = np.sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2)
        reprojection[reprojection < 10] = 10
        p[:, :, 0] = p[:, :, 0] / reprojection
        p[:, :, 1] = p[:, :, 1] / reprojection

        grad_v[:, :, 0] = dx(v_[:, :, 0])
        grad_v[:, :, 1] = dy(v_[:, :, 1])
        grad_v[:, :, 2] = dy(v_[:, :, 0])
        grad_v[:, :, 3] = dx(v_[:, :, 1])

        q = q + alpha1 * sigma / eta_q * grad_v

        reproject = np.sqrt(q[:, :, 0] ** 2 + q[:, :, 1] ** 2 + q[:, :, 2] ** 2 + q[:, :, 3] ** 2)
        reproject[reproject < 10] = 10
        q[:, :, 0] = q[:, :, 0] / reproject
        q[:, :, 1] = q[:, :, 1] / reproject
        q[:, :, 2] = q[:, :, 2] / reproject
        q[:, :, 3] = q[:, :, 3] / reproject

        u_ = u
        v_ = v

        div_p = dxz(grad_i * p[:, :, 0] + grad_k * p[:, :, 1]) + dyz(grad_k * p[:, :, 0] + grad_j * p[:, :, 1])

        tau_eta_u = tau / eta_u
        u = (u_ + tau_eta_u * (alpha2 * div_p + dw)) / (1 + tau_eta_u * weight)
        qw_x = np.append(np.zeros_like(q[:, -1, 0].reshape((q.shape[0], 1))),
                         q[:, :-1, 0].reshape((q.shape[0], q.shape[1] - 1)), axis=1)
        qw_w = np.append(np.zeros_like(q[:, -1, 3].reshape((q.shape[0], 1))),
                         q[:, :-1, 3].reshape((q.shape[0], q.shape[1] - 1)), axis=1)
        qn_y = np.append(np.zeros_like(q[-1, :, 1].reshape((1, q.shape[1]))),
                         q[:-1, :, 1].reshape((q.shape[0] - 1, q.shape[1])), axis=0)
        qn_z = np.append(np.zeros_like(q[-1, :, 2].reshape((1, q.shape[1]))),
                         q[:-1, :, 2].reshape((q.shape[0] - 1, q.shape[1])), axis=0)

        div_q1 = (np.append(q[:, :-1, 0].reshape((q.shape[0], q.shape[1] - 1)),
                            np.zeros_like(q[:, -1, 0].reshape((q.shape[0], 1))), axis=1) - qw_x) + (
                         np.append(q[:-1, :, 2].reshape((q.shape[0] - 1, q.shape[1])),
                                   np.zeros_like(q[-1, :, 2].reshape((1, q.shape[1]))), axis=0) - qn_z)
        div_q2 = (np.append(q[:, :-1, 3].reshape((q.shape[0], q.shape[1] - 1)),
                            np.zeros_like(q[:, -1, 3].reshape((q.shape[0], 1))), axis=1) - qw_w) + (
                         np.append(q[:-1, :, 1].reshape((q.shape[0] - 1, q.shape[1])),
                                   np.zeros_like(q[-1, :, 1].reshape((1, q.shape[1]))), axis=0) - qn_y)

        dq1 = grad_i * p[:, :, 0] + grad_k * p[:, :, 1]
        dq2 = grad_k * p[:, :, 0] + grad_j * p[:, :, 1]

        div_q = np.stack((div_q1, div_q2), axis=2)
        dq = np.stack((dq1, dq2), axis=2)
        v = v_ + tau / eta_v * (alpha2 * dq + alpha1 * div_q)
        u_ = u + mu * (u - u_)
        v_ = v + mu * (v - v_)

        sigma = sigma / mu
        tau = tau * mu

    return u * 255.


def g(x, K=5):
    return 1.0 / (1.0 + ((x * x) / (K * K)))


def c(I, K=5):
    cv = g(I[1:, :] - I[:-1, :], K)
    ch = g(I[:, 1:] - I[:, :-1], K)

    return cv, ch


def diffuse_step(cv, ch, I, l=0.24):
    dv = I[1:, :] - I[:-1, :]
    dh = I[:, 1:] - I[:, :-1]

    tv = l * cv * dv  # vertical transmissions
    I[1:, :] -= tv
    I[:-1, :] += tv
    del (dv, tv)

    th = l * ch * dh  # horizontal transmissions
    I[:, 1:] -= th
    I[:, :-1] += th
    del (dh, th)

    return I


def anisotropic_diffusion(I1, I2, I, N=5, l=0.24, K=5):
    I1 = I1 / 255.
    I2 = I2 / 255.
    I = I / 255.
    for i in tqdm(range(N)):
        cv1, ch1 = c(I1, K=K)
        I1 = diffuse_step(cv1, ch1, I1, l=l)

        cv2, ch2 = c(I2, K=K)
        I2 = diffuse_step(cv2, ch2, I2, l=l)
        cv = np.minimum(cv1, cv2)
        ch = np.minimum(ch1, ch2)
        del (cv1, ch1, cv2, ch2)
        I = diffuse_step(cv, ch, I, l=l)

        del (cv, ch)

    return I * 255.
