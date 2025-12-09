# gen_input_random3d.py
import sys
import numpy as np

# 要跟 C 程式那邊一致
TOTAL_TIME = 20.0
DT = 0.01
DUMP_INTERVAL = 10

# 假設 C 端 G = 1.0
G = 1.0

# 空間半徑範圍：粒子會在這個球殼內隨機分布
R_MIN = 5.0
R_MAX = 50.0

# 質量分佈
M_MIN = 0.5
M_MAX = 5.0

# 速度 noise 程度
VEL_NOISE = 0.2

# 是否有中心大質量 (0 表示沒有)
CENTRAL_MASS = 500.0  # 想關掉就設 0.0


def sample_mass():
    # 你可以改成 np.random.lognormal(...) 之類的
    return np.random.uniform(M_MIN, M_MAX)


def random_unit_vector():
    """在 3D 單位球面上均勻取一個方向"""
    z = 2.0 * np.random.rand() - 1.0  # cos(theta) in [-1,1]
    t = 2.0 * np.pi * np.random.rand()  # phi in [0, 2pi)
    r_xy = np.sqrt(1.0 - z * z)
    x = r_xy * np.cos(t)
    y = r_xy * np.sin(t)
    return np.array([x, y, z], dtype=float)


def gen_random3d(N, filename):
    bodies = []

    # (可選) 中心大質量在原點
    if CENTRAL_MASS > 0.0:
        bodies.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CENTRAL_MASS])

    # 其他 N 顆粒子
    for _ in range(N):
        # 半徑：用 cbrt(u) -> 均勻分布在體積
        u = np.random.rand()
        r = np.cbrt(u) * (R_MAX - R_MIN) + R_MIN

        # 隨機方向
        dir_r = random_unit_vector()
        pos = r * dir_r
        x, y, z = pos

        m = sample_mass()

        # 如果有中心 mass，就給「接近圓軌道」的切線速度
        if CENTRAL_MASS > 0.0:
            # 根據中心質量估計圓軌道速度
            v_c = np.sqrt(G * CENTRAL_MASS / r)

            # 找一個與 dir_r 不共線的向量，做出切線方向
            tmp = random_unit_vector()
            if abs(np.dot(tmp, dir_r)) > 0.9:
                tmp = random_unit_vector()
            # 切線方向 = cross(r, tmp)
            v_dir = np.cross(dir_r, tmp)
            v_dir /= np.linalg.norm(v_dir)

            v = v_c * v_dir

            # 加一點 noise
            v += VEL_NOISE * np.random.randn(3)
        else:
            # 沒有中心質量：純隨機小速度
            v = VEL_NOISE * np.random.randn(3)

        vx, vy, vz = v

        bodies.append([x, y, z, vx, vy, vz, m])

    N_total = len(bodies)

    with open(filename, "w") as f:
        f.write(f"{N_total}\n")
        f.write(f"{TOTAL_TIME} {DT} {DUMP_INTERVAL}\n")
        for b in bodies:
            f.write(
                "{:.6f} {:.6f} {:.6f} " "{:.6f} {:.6f} {:.6f} " "{:.6f}\n".format(*b)
            )

    print(f"Generated {N_total} bodies -> {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_input_random3d.py N output.txt")
    else:
        N = int(sys.argv[1])
        out = sys.argv[2]
        gen_random3d(N, out)
