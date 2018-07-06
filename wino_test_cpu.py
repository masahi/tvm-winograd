import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi import util
from topi.nn import pad
import sys

#sys.setrecursionlimit(10000)

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
    with tvm.build_config(auto_unroll_max_step=500,
                          unroll_explicit=True):
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

def decl_U(data, kernel, stride, padding, out_dtype):
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

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

    bna, bnb = 8, 8
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    # transform kernel
    G = const_array(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((K // bna, C // bnb, alpha, alpha, bna, bnb), lambda k, c, eps, nu, kk, cc:
                    tvm.sum(kernel[k * bna + kk][c * bnb + c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')
    outs = [U]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    U = op.output(0)
    kernel, G = s[U].op.input_tensors
    s[G].compute_inline()
    k, c, eps, nu, kk, cc = s[U].op.axis
    r_kh, r_kw = s[U].op.reduce_axis
    s[U].reorder(k, c, kk, cc, eps, nu, r_kh, r_kw)
    _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
    kk = s[U].fuse(kk, cc)
    s[U].vectorize(kk)
    fused = s[U].fuse(k, c)
    s[U].parallel(fused)
    
    return s

def decl_V(data, kernel,  stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    K = K

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 8, 8
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")    
    input_tile = tvm.compute((P_round // bnb, C // bna, alpha, alpha, bnb, bna),
                             lambda b, c, eps, nu, bb, cc:
                             tvm.select(b * bnb + bb < P,\
                             data_pad[(b*bnb+bb) // (nH*nW)][c*bna + cc][(b*bnb+bb) // nW % nH * m + eps]\
                             [(b*bnb+bb) % nW * m + nu], tvm.const(0, data_pad.dtype)),
                             name='d')
    
    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P_round // bnb, C // bna, alpha, alpha, bnb, bna), lambda b, c, eps, nu, bb, cc:
                    tvm.sum(input_tile[b][c][r_eps][r_nu][bb][cc] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')
    outs = [V]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    V = op.output(0)
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    s[data_pad].compute_inline()
    s[B].compute_inline()
    b, c, eps, nu, bb, cc = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, cc, eps, nu, r_nu, r_eps)
    # _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    bb = s[V].fuse(bb, cc)
    s[V].vectorize(bb)
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)
    s[d].compute_at(s[V], fused)
    b, c, eps, nu, bb, cc = s[d].op.axis
    s[d].unroll(bb)
    s[d].unroll(cc)
    return s
    

def decl_M(data, kernel, U, V, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    m = 2
    r = 3
    alpha = m + r - 1
    K = K

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 8, 8
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((K // bna, P_round //bnb, alpha, alpha, bna, bnb), lambda k, b, eps, nu, kk, bb:
                    tvm.sum(U[k][c // bnb][eps][nu][k % bna][c % bnb] *
                            V[b][c // bna][eps][nu][b % bnb][c % bna], axis=c), name='M')
    outs = [M]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    M = op.output(0)
    bna, bnb = 8, 8
    k, b, eps, nu, kk, bb = s[M].op.axis
    c = s[M].op.reduce_axis[0]

    #MM = s.cache_write(M, 'global')
    yo, xo, yi, xi = k, b, kk, bb
    fused = s[M].fuse(yo, xo, eps)
    s[M].parallel(fused)
    #s[MM].compute_at(s[M], fused)
    co, ci = s[M].split(c, factor=4)
    #_, _, yi, xi = s[MM].op.axis
    s[M].reorder(co, yi, ci, xi)
    #s[M].reorder(c, yi, xi)
    s[M].unroll(ci)
    s[M].vectorize(xi)

    return s


def decl_output(data, kernel, M, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    K = K

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 8, 8
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0


    # inverse transform
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                    tvm.sum(M[k//bna][(n * nH * nW + (h//m) * nW + w//m)//bna][r_eps][r_nu][k%bna][(n * nH * nW + (h//m) * nW + w//m)%bna] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')
    outs = [output]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    _, A = s[output].op.input_tensors
    s[A].compute_inline()
    n, k, h, w = s[output].op.axis
 #   output_L = s.cache_write(output, "global")
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(n, k, ho, wo, hi, wi)
    fused = s[output].fuse(n, k, ho, wo)
    s[output].parallel(fused)
    
    return s
    

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

    bna, bnb = 8, 8
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
    U = tvm.compute((K // bna, C, alpha, alpha, bna), lambda k, c, eps, nu, kk:
                    tvm.sum(kernel[k * bna + kk][c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P_round // bnb, C, alpha, alpha, bnb), lambda  b, c, eps, nu, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P_round), lambda eps, nu, k, b:
                    tvm.sum(U[k // bna][c][eps][nu][k % bna] *
                            V[b // bnb][c][eps][nu][b % bnb], axis=c), name='M')

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
#    Y = op.input_tensors[0]
    M, A = s[output].op.input_tensors
    U, V = s[M].op.input_tensors
    kernel, G = s[U].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    s[data_pad].compute_inline()

    # pack input tiles
    c, b, eps, nu, bb = s[d].op.axis
    s[d].unroll(bb)

    # transform kernel
    s[G].compute_inline()
    k, c, eps, nu, kk, = s[U].op.axis
    r_kh, r_kw = s[U].op.reduce_axis
    s[U].reorder(k, c, kk, eps, nu, r_kh, r_kw)
    _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
    s[U].vectorize(kk)
    fused = s[U].fuse(k, c)
    s[U].parallel(fused)

    # transform image
    s[B].compute_inline()
    b, c, eps, nu, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, eps, nu, r_nu, r_eps)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    s[V].vectorize(bb)
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)
    s[d].compute_at(s[V], fused)    

    # batch gemm
    bna, bnb = 8, 8
    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    #MM = s.cache_write(M, 'global')
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    fused = s[M].fuse(eps, nu, yo, xo)
    s[M].parallel(fused)
    #s[MM].compute_at(s[M], fused)
    co, ci = s[M].split(c, factor=4)
    #_, _, yi, xi = s[MM].op.axis
    s[M].reorder(co, yi, ci, xi)
    #s[M].reorder(c, yi, xi)
    s[M].unroll(ci)
    s[M].vectorize(xi)

    # inverse transform
    s[A].compute_inline()
    n, k, h, w = s[output].op.axis
 #   output_L = s.cache_write(output, "global")
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(n, k, ho, wo, hi, wi)
    fused = s[output].fuse(n, k, ho, wo)
    s[output].parallel(fused)
#    s[output_L].compute_at(s[output], fused)

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


def test_components(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size
    m = 2
    r = 3
    alpha = m + r - 1
    K = num_filter
    H = W = in_size
    N = batch
    C = in_channel

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 8, 8
    P_round = (P + bnb - 1) // bnb * bnb

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((K // bna, C // bnb, alpha, alpha, bna, bnb), name='U')
    V = tvm.placeholder((P_round // bnb, C // bna, alpha, alpha, bnb, bna), name='V')
    M = tvm.placeholder((K // bnb, P_round // bna, alpha, alpha, bnb, bna ), name='M')
    output = tvm.placeholder((N, K, in_size, in_size), name='output')

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
    u_np = np.zeros(util.get_const_tuple(U.shape), dtype=dtype)
    v_np = np.zeros(util.get_const_tuple(V.shape), dtype=dtype)
    m_np = np.zeros(util.get_const_tuple(M.shape), dtype=dtype)
    output_np = np.zeros(util.get_const_tuple(output.shape), dtype=dtype)

    with tvm.target.create(device):
        s_U = decl_U(A, W, stride, padding, dtype)
        s_V = decl_V(A, W, stride, padding, dtype)
        s_M = decl_M(A, W, U, V, stride, padding, dtype)
        s_output = decl_output(A, W, M, stride, padding, dtype)        
        
    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    v = tvm.nd.array(v_np, ctx)
    m = tvm.nd.array(m_np, ctx)
    output_tvm = tvm.nd.array(output_np, ctx)        
    num_runs = 100
    times = {}
    with tvm.build_config(auto_unroll_max_step=500,
                          unroll_explicit=True):
        func = tvm.build(s_U, [A, W, U], device)
        func(a, w, u)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        times["U"] = timer(a, w, u).mean * 1000
        
        func = tvm.build(s_V, [A, W, V], device)
        func(a, w, v)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        times["V"] = timer(a, w, v).mean * 1000
        #print(tvm.lower(s_V, [A, W, V], simple_mode=True))
        
        func = tvm.build(s_M, [A, W, U, V, M], device)
        func(a, w, u, v, m)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        times["M"] = timer(a, w, u, v, m).mean * 1000
        #print(tvm.lower(s_M, [A, W, U, V, M], simple_mode=True))

        func = tvm.build(s_output, [A, W, M, output], device)
        func(a, w, m, output_tvm)
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        times["output"] = timer(a, w, m, output_tvm).mean * 1000
        #print(tvm.lower(s_output, [A, W, M, output], simple_mode=True))        
    return times


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
    with tvm.build_config(auto_unroll_max_step=500,
                          unroll_explicit=True):
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        #print(tvm.lower(s, [A, W, B], simple_mode=True))
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


workloads1 = [(1, 128, 122, 128),
             (1, 64, 56, 64),
             (1, 64, 64, 32),
             (1, 64, 224, 64),
             (1, 64, 112, 128),
             (1, 512, 28, 512),
             (1, 128, 28, 128),
             (1, 256, 14, 256),
            ]

workloads2 = [# (1, 3, 128, 32),
              # (1, 32, 128, 16),
              # (1, 16, 128, 8),
              # (1, 8, 128, 16),
              # (1, 16, 128, 32),
              # (1, 32, 64, 32),
              # (1, 32, 64, 64),
              # (1, 64, 32, 64),
              # (1, 64, 16, 64),
              # (1, 64, 8, 64),
              # (1, 128, 16, 64),
              # (1, 128, 32, 64),
              # (1, 96, 64, 32),
              # (1, 40, 128, 16),
              (1, 16, 128, 16)
             ]

workloads3 = [(10, 3, 128, 32),
              (10, 32, 128, 16),
              (10, 16, 128, 8),
              (10, 8, 128, 16),
              (10, 16, 128, 32),
              (10, 32, 64, 32),
              (10, 32, 64, 64),
              (10, 64, 32, 64),
              (10, 64, 16, 64),
              (10, 64, 8, 64),
              (10, 128, 16, 64),
              (10, 128, 32, 64),
              (10, 96, 64, 32),
              (10, 40, 128, 16),
              (10, 16, 128, 16)
             ]

wino_times = []
direct_times = []
device = "llvm"
workloads = workloads2

for workload in workloads:
    times = test_components(*workload, 3, 1, 1, device)
#     t_wino = test_winograd(*workload, 3, 1, 1, device)
#     wino_times.append(t_wino * 1000)    
#     t_direct = reference_direct(*workload, 3, 1, 1, device)
#     direct_times.append(t_direct * 1000)
    
    print("Workload: ", workload)    
    for (k,v) in times.items():
        print("%s: %f" % (k, v))
#     print("Total: %f" % np.sum(list(times.values())))
#     print("Wino time: ", wino_times[-1])    
#     print("Direct: %f\n" % direct_times[-1])


# generate_table(workloads, wino_times, direct_times)
