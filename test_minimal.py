import tvm
import numpy as np

def decl_V(A):
    temp_expr = {}
    for j in range(4):
        temp_expr[(0, j)] = A[0][j] - A[2][j]
        temp_expr[(1, j)] = A[1][j] + A[2][j]
        temp_expr[(2, j)] = A[2][j] - A[1][j]
        temp_expr[(3, j)] = A[1][j] - A[3][j]

    def compute_temp(i, j):
        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    T1 = tvm.compute((4,4), compute_temp, name="T1")

    v_expr = {}
    for i in range(4):
        v_expr[(i, 0)] = T1[i][0] - T1[i][2]
        v_expr[(i, 1)] = T1[i][1] + T1[i][2]
        v_expr[(i, 2)] = T1[i][2] - T1[i][1]
        v_expr[(i, 3)] = T1[i][1] - T1[i][3]

    def compute_V(i, j):
        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((4,4), compute_V)

    return V

def decl_output(M):
    temp_expr = {}
    for j in range(4):
        t0 = M[0][j] + M[1][j]
        t1 = M[1][j] - M[2][j]
        temp_expr[(0,j)] = t0 + M[2][j]
        temp_expr[(1,j)] = t1 - M[3][j]

    def compute_temp(i, j):
        now = tvm.const(0.0, "float32")
        for ii in range(2):
            for jj in range(4):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    T1 = tvm.compute((2,4), compute_temp, name="T1")

    output_expr = {}
    for i in range(2):
        t0 = T1[i][0] + T1[i][1]
        t1 = T1[i][1] - T1[i][2]
        output_expr[(i,0)] = t0 + T1[i][2]
        output_expr[(i,1)] = t1 - T1[i][3]

    def compute_output(i, j):
        now = tvm.const(0.0, "float32")
        for ii in range(2):
            for jj in range(2):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 output_expr[(ii, jj)],
                                 now)
        return now

    output = tvm.compute((2,2), compute_output)

    return output

def schedule(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    T1 = s[output].op.input_tensors[0]
    i, j = s[output].op.axis
    s[output].unroll(i)
    s[output].unroll(j)
    i, j = s[T1].op.axis
    s[T1].unroll(i)
    s[T1].unroll(j)

    return s

A = tvm.placeholder((4, 4), name="A")
M = tvm.placeholder((4, 4), name="M")
device = "llvm"
with tvm.target.create(device):
    T = decl_V(A)
    s = schedule([T])
    output = decl_output(M)
    s2 = schedule([output])

#print(tvm.lower(s, [A, T], simple_mode=True))
func = tvm.build(s, [A, T], device)

ctx = tvm.context(device, 0)
a_np = np.random.uniform(size=(4,4)).astype("float32")
t_np = np.random.uniform(size=(4,4)).astype("float32")
a = tvm.nd.array(a_np, ctx)
t = tvm.nd.array(t_np, ctx)

func(a,t)
#print(t)

B_data = np.array([
    [1, 0, 0, 0],
    [0, 1, -1, 1],
    [-1, 1, 1, 0],
    [0, 0, 0, -1]
], "float32")

ref = np.dot(np.dot(B_data.transpose(), a_np), B_data)
# print(ref)

print(tvm.lower(s2, [M, output], simple_mode=True))
func = tvm.build(s2, [M, output], device)

m_np = np.random.uniform(size=(4,4)).astype("float32")
output_np = np.random.uniform(size=(2,2)).astype("float32")
m = tvm.nd.array(m_np, ctx)
output = tvm.nd.array(output_np, ctx)

func(m, output)

A_data = np.array([
    [1, 0],
    [1, 1],
    [1, -1],
    [0, -1],
], "float32")

ref = np.dot(np.dot(A_data.transpose(), m_np), A_data)
print(np.max(np.abs(output.asnumpy() - ref)))

