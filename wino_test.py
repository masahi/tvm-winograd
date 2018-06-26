import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.nn.util import *
from topi.util import *
from topi import util
from topi.nn import pad
from topi import tag

def const_array(data, name):
    row, col = data.shape
    dtype = str(data.dtype)

    def select_array(i, j):
        now = tvm.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.select(tvm.all(i % row == ii, j % col == jj),
                                 tvm.const(data[ii][jj], dtype),
                                 now)
        return now
    return tvm.compute(data.shape, select_array, name=name)

def decl_winograd(data, U, stride, padding, out_dtype):
    N, CI, H, W = [util.get_const_int(x) for x in data.shape]
    _, _, CO, CI = [util.get_const_int(x) for x in U.shape]
    HPAD, WPAD = 1,1
    HSTR, WSTR = 1,1

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)
    B = const_array(B_data, 'B')

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)
    A = const_array(A_data, 'A')

    m = 2
    r = 3
    alpha = m + r - 1
    K = CO
    C = CI

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    # pack input tile
    input_tile = tvm.compute((C, P, alpha, alpha),
                             lambda c, b, eps, nu:
                             tvm.select(b < P, data_pad[b // (nH*nW)][c][b// nW % nH * m + eps][b % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    # transform image

    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P, C), lambda eps, nu, b, c:
                    tvm.sum(input_tile[c][b][r_eps][r_nu] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][k][c] *
                            V[eps][nu][b][c], axis=c), name='M')

    # inverse transform and unpack
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                    tvm.sum(M[r_eps][r_nu][k][n * nH * nW + (h//m) * nW + w//m] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')
    return output

def schedule_winograd(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)

    M, A = s[output].op.input_tensors
    U, V = s[M].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    s[data_pad].compute_inline()

    num_thread = 16
    s[d].compute_inline()

    # transform image
    s[B].compute_inline()
    r_eps, r_nu = s[V].op.reduce_axis
    eps, nu, p, c = s[V].op.axis
    s[V].reorder(eps, nu, p, c, r_nu, r_eps)
    po, pi = s[V].split(p, factor=num_thread)
    co, ci = s[V].split(c, factor=num_thread)
    s[V].reorder(eps, nu, po, co, pi, ci)
    fused = s[V].fuse(eps, nu, po, co)
    s[V].bind(pi, tvm.thread_axis("threadIdx.y"))
    s[V].bind(ci, tvm.thread_axis("threadIdx.x"))
    s[V].bind(fused, tvm.thread_axis("blockIdx.x"))

    eps, nu, k, p = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    ko, ki = s[M].split(k, factor=num_thread)
    po, pi = s[M].split(p, factor=num_thread)
    z = s[M].fuse(eps, nu)
    s[M].bind(ki, tvm.thread_axis("threadIdx.y"))
    s[M].bind(pi, tvm.thread_axis("threadIdx.x"))
    s[M].bind(ko, tvm.thread_axis("blockIdx.y"))
    s[M].bind(po, tvm.thread_axis("blockIdx.x"))
    s[M].bind(z, tvm.thread_axis("blockIdx.z"))

    # inverse transform
    s[A].compute_inline()
    n, k, h, w = s[output].op.axis
    ho, hi = s[output].split(h, factor=num_thread)
    wo, wi = s[output].split(w, factor=num_thread)
    s[output].reorder(k, ho, wo, hi, wi)
    fused = s[output].fuse(k, ho, wo)
    s[output].bind(hi, tvm.thread_axis("threadIdx.y"))
    s[output].bind(wi, tvm.thread_axis("threadIdx.x"))
    s[output].bind(fused, tvm.thread_axis("blockIdx.x"))

    return s

def transform_filter(w_np):
    num_filter, in_channel, kernel, kernel = w_np.shape
    G = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], w_np.dtype)

    out = np.empty((4, 4, num_filter, in_channel), w_np.dtype)
    for i in range(num_filter):
        for j in range(in_channel):
            out[:, :, i, j] = np.dot(G, np.dot(w_np[i, j], G.transpose()))
    return out

def test(batch, in_channel, in_size, num_filter, device):
    in_height = in_width = in_size
    kernel = 3
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((4, 4, num_filter, in_channel), name='U')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1
    stride = 1
    padding = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.reference_direct")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()
    u_np = transform_filter(w_np)

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(b_np.shape), dtype=dtype), ctx)

    with tvm.target.create(device):
        B = topi.nn.conv2d(A, W, stride, padding, layout='NCHW')
        s = topi.generic.schedule_conv2d_nchw([B])
        B_wino = decl_winograd(A, U, stride, padding, dtype)
        s_wino = schedule_winograd([B_wino])

    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda")):
        num_runs = 100
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        t_direct = timer(a, w, b).mean

        func = tvm.build(s_wino, [A, U, B_wino], device)
        func(a, u, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        t_wino = timer(a, u, b).mean
        #print(tvm.lower(s_wino, [A, U, B_wino], simple_mode=True))
        return t_wino, t_direct

device = "cuda"
workload = (1, 64, 224, 64)
t_wino, t_direct = test(*workload, device)
print("Winograd: %f msec, Reference: %f msec" % (t_wino, t_direct))
