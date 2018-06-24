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

def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Running on target: %s" % device)
    with tvm.target.create(device):
        dW = topi.nn.dilate(W, (1, 1, dilation, dilation))
        B = topi.nn.conv2d(A, dW, stride, padding, layout='NCHW')
        s1 = topi.generic.schedule_conv2d_nchw([B])
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda")):
        func = tvm.build(s1, [A, W, B], device, name="conv2d_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
        #print(tvm.lower(s1, [A, W, B], simple_mode=True))
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        t = timer(a, w, b).mean
        print("elapsed %f" % t)

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

def decl_winograd(data, U, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, CI, H, W = [util.get_const_int(x) for x in data.shape]
    _, _, CO, CI = [util.get_const_int(x) for x in U.shape]
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

    m = 2
    r = 3
    alpha = m + r - 1
    K = CO
    C = CI

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = tvm.compute((C, P_round, alpha, alpha),
                             lambda c, b, eps, nu:
                             tvm.select(b < P, data_pad[b // (nH*nW)][c][b// nW % nH * m + eps][b % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P_round, C), lambda eps, nu, b, c:
                    tvm.sum(input_tile[c][b][r_eps][r_nu] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P_round), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][k][c] *
                            V[eps][nu][b][c], axis=c), name='M')

    # inverse transform and unpack
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                    tvm.sum(M[r_eps][r_nu][k][n * nH * nW + (h//m) * nW + w//m] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')

    return output

def schedule_winograd(s, op):
    output = op.output(0)

    M, A = s[output].op.input_tensors
    U, V = s[M].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    s[data_pad].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, "local")

    eps, nu, P, C = s[V].op.axis
    r_eps, r_nu = s[VL].op.reduce_axis
    s[V].reorder(P, C, eps, nu)

    ho, hi = s[V].split(P, factor=16)
    wo, wi = s[V].split(C, factor=16)
    s[V].bind(hi, tvm.thread_axis("threadIdx.y"))
    s[V].bind(wi, tvm.thread_axis("threadIdx.x"))
    s[V].bind(ho, tvm.thread_axis("blockIdx.y"))
    s[V].bind(wo, tvm.thread_axis("blockIdx.x"))

    s[VL].compute_at(s[V], wi)
    s[d].compute_at(s[V], wi)

    # batch gemm
    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2

    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    ML = s.cache_write(M, "local")
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    z = s[M].fuse(eps, nu)

    yoo, yoi = s[M].split(yo, 16)
    xoo, xoi = s[M].split(xo, 16)
    s[M].bind(z, tvm.thread_axis("blockIdx.z"))
    s[M].bind(yoo, tvm.thread_axis("blockIdx.y"))
    s[M].bind(yoi, tvm.thread_axis("threadIdx.y"))
    s[M].bind(xoo, tvm.thread_axis("blockIdx.x"))
    s[M].bind(xoi, tvm.thread_axis("threadIdx.x"))
    s[ML].compute_at(s[M], xoi)

    # inverse transform
    s[A].compute_inline()

    # schedule output
    if output.op in s.outputs:  # no bias
        output = output
    else:                       # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, k, h, w = s[output].op.axis
    output_L = s.cache_write(output, "local")
    ho, hi = s[output].split(h, factor=16)
    wo, wi = s[output].split(w, factor=16)
    s[output].reorder(k, ho, wo, hi, wi)
    s[output].bind(hi, tvm.thread_axis("threadIdx.y"))
    s[output].bind(wi, tvm.thread_axis("threadIdx.x"))
    fused = s[output].fuse(k, ho, wo)
    s[output].bind(fused, tvm.thread_axis("blockIdx.x"))
    s[output_L].compute_at(s[output], wi)

def schedule_conv2d_nchw(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        # if tag.is_broadcast(op.tag):
        #     if op not in s.outputs:
        #         s[op].compute_inline()
        #     for tensor in op.input_tensors:
        #         if tensor.op.input_tensors:
        #             traverse(tensor.op)
        schedule_winograd(s, op)

    traverse(outs[0].op)
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


def test(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((4, 4, num_filter, in_channel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    with tvm.target.create(device):
        B = decl_winograd(A, U, stride, padding, dtype)
        s = schedule_conv2d_nchw([B])

    u_np = transform_filter(w_np)

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda")):
        func = tvm.build(s, [A, U, B], device)
        #print(tvm.lower(s, [A, U, B], simple_mode=True))
        func(a, u, b)
        num_runs = 10
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        t = timer(a, u, b).mean
        print("elapsed %f" % t)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        #print(func.imported_modules[0].get_source())

device = "cuda"

test(1, 128, 122, 128, 3, 1, 1, 1, device)
# test(1, 64, 64, 32, 3, 1, 1, 1, device)
# verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1, 1, device)
# verify_conv2d_nchw(1, 64, 64, 32, 3, 1, 1, 1, device)
