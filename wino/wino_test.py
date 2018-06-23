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

def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi

def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))

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
    N, CI, H, W = [util.get_const_int(x) for x in data.shape]
    CO, CI, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1 and KH == 3 and KW == 3
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    G_data = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
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
    Y = tvm.compute((K, P, m, m), lambda k, b, vh, vw:
                    tvm.sum(M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # unpack output
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                         Y[k][n * nH * nW + (h//m) * nW + w//m][h % m][w % m]
                         # thw following term is used to make the padding effective,
                         # otherwise the padding will be eliminated by bound inference
                         + tvm.const(0, out_dtype) * M[alpha-1][alpha-1][K-1][P_round-1],
                         name='output', tag='winograd_conv_output')

    return output

def schedule_winograd(s, op):
    output = op.output(0)

    Y = op.input_tensors[0]
    M, A = s[Y].op.input_tensors
    U, V = s[M].op.input_tensors
    kernel, G = s[U].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    # dilation
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    # padding
    s[data_pad].compute_inline()

    # pack input tiles
    c, b, eps, nu, bb = s[d].op.axis
    s[d].reorder(eps, nu, bb)
    aha = s[d].fuse(eps, nu)
    s[d].unroll(bb)
    tile_and_bind3d(s, d, c, b, aha, 4, 1, 1)

    # transform kernel
    s[G].compute_inline()
    eps, nu, k, c, kk, = s[U].op.axis
    r_kh, r_kw = s[U].op.reduce_axis
    s[U].reorder(k, c, kk, eps, nu, r_kh, r_kw)
    _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
    tile_and_bind(s, U, k, c, 1, 256)

    # transform image
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, eps, nu, r_nu, r_eps)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    tile_and_bind(s, V, b, c, 2, 1)

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
    z = s[M].fuse(eps, nu)
    tile_and_bind3d(s, M, z, yo, xo, 1, 8, 1)

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    _ = [s[Y].unroll(x) for x in [vh, vw, r_eps, r_nu]]
    tile_and_bind(s, Y, k, b, 4, 1)

    # schedule output
    if output.op in s.outputs:  # no bias
        output = output
    else:                       # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, k, h, w = s[output].op.axis
    tile_and_bind3d(s, output, k, h, w, 1, 2, 2)

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

def test(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
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

    device = "cuda"
    with tvm.target.create(device):
        B = decl_winograd(A, W, stride, padding, dtype)
        s = schedule_conv2d_nchw([B])

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda")):
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        print(func.imported_modules[0].get_source())
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)


test(1, 128, 122, 128, 3, 1, 1)
