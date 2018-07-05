import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi import util
from topi.nn import pad

def reference_direct(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.reference_direct")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    with tvm.target.create(device):
        B = topi.nn.conv2d(A, W, stride, padding, layout='NCHW')
        s1 = topi.generic.schedule_conv2d_nchw([B])
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=False):
        func = tvm.build(s1, [A, W, B], device)
        #print(tvm.lower(s1, [A, W, B], simple_mode=True))
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        return timer(a, w, b).mean

def const_array(data, name):
    """ convert an const array to tvm tensor"""
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

def decl_winograd(data, kernel, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)

    G_data = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    K = K

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = tvm.compute((C, P_round // bnb, alpha, alpha, bnb),
                             lambda c, b, eps, nu, bb:
                             tvm.select(b * bnb + bb < P,\
                             data_pad[(b*bnb+bb) // (nH*nW)][c][(b*bnb+bb) // nW % nH * m + eps]\
                             [(b*bnb+bb) % nW * m + nu], tvm.const(0, data_pad.dtype)),
                             name='d')

    # transform kernel
    G = const_array(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((alpha, alpha, K // bna, C, bna), lambda eps, nu, k, c, kk:
                    tvm.sum(kernel[k * bna + kk][c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P_round // bnb, C, bnb), lambda eps, nu, b, c, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P_round), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][k // bna][c][k % bna] *
                            V[eps][nu][b // bnb][c][b % bnb], axis=c), name='M')

    # inverse transform
    A = const_array(A_data, 'A')
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
    kernel, G = s[U].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    s[data_pad].compute_inline()

    # pack input tiles
    c, b, eps, nu, bb = s[d].op.axis
    s[d].reorder(eps, nu, bb)
    s[d].unroll(bb)
    fused = s[d].fuse(c, b)
    s[d].parallel(fused)

    # transform kernel
    s[G].compute_inline()
    eps, nu, k, c, kk, = s[U].op.axis
    r_kh, r_kw = s[U].op.reduce_axis
    s[U].reorder(k, c, kk, eps, nu, r_kh, r_kw)
    _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
    s[U].vectorize(kk)
    fused = s[U].fuse(k, c)
    s[U].parallel(fused)

    # transform image
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, eps, nu, r_nu, r_eps)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    s[V].vectorize(bb)
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)

    # batch gemm
    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2

    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    s[M].reorder(c, yi, xi)
    c, c_unroll = s[M].split(c, 2)
    s[M].unroll(c_unroll)
    s[M].unroll(yi)
    s[M].vectorize(xi)
    z = s[M].fuse(eps, nu)
    s[M].parallel(z)

    # inverse transform
    s[A].compute_inline()
    n, k, h, w = s[output].op.axis
    fused = s[output].fuse(n, k)
    s[output].parallel(fused)

    return s


def transform_filter(w_np):
    num_filter, in_channel, kernel, kernel = w_np.shape
    G = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], w_np.dtype)

    out = np.empty((4, 4, in_channel, num_filter), w_np.dtype)
    for i in range(in_channel):
        for j in range(num_filter):
            out[:, :, i, j] = np.dot(G, np.dot(w_np[j, i], G.transpose()))
    return out


def test_winograd(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.wino")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    with tvm.target.create(device):
        B = decl_winograd(A, W, stride, padding, dtype)
        s = schedule_winograd([B])

    u_np = transform_filter(w_np)

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=False):
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        return timer(a, w, b).mean

# for copy paste as markdown
def generate_table(workloads, wino_times, direct_times):
    print("| (batch,CI,size,CO) | TVM Winograd (This code) | TVM Direct |")
    print("|------------- |:-------------:|:-------------:|")
    for (workload, t_wino, t_direct) in zip(workloads, wino_times, direct_times):
        print("|", workload, "| %.3f | %.3f |" % (t_wino,  t_direct))


workloads = [(1, 128, 122, 128),
             (1, 64, 56, 64),
             (1, 64, 64, 32),
             (1, 64, 224, 64),
             (1, 64, 112, 128),
             (1, 512, 28, 512),
             (1, 128, 28, 128),
             (1, 256, 14, 256),
            ]

wino_times = []
direct_times = []
device = "llvm"

for workload in workloads:
    print(workload)
    print("wino")
    t_wino = test_winograd(*workload, 3, 1, 1, device)
    print("direct")
    t_direct = reference_direct(*workload, 3, 1, 1, device)

    wino_times.append(t_wino * 1000)
    direct_times.append(t_direct * 1000)

generate_table(workloads, wino_times, direct_times)
