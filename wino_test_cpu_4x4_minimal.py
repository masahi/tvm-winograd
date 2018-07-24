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

def decl_V(data, kernel,  stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")    
    # transform image
    def compute_temp(b, c, eps, nu, bb, cc):
        temp_expr = {}
        batch_index = (b*bnb+bb) // (nH*nW)
        h = (b*bnb+bb) // nW % nH * m
        w = (b*bnb+bb) % nW * m
        for j in range(6):
            t0 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]*4.0
            t1 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]*4.0
            t2 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]
            t3 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]
            
            temp_expr[(0, j)] = data_pad[batch_index][c][h+0][w+j][cc] * 4.0 - data_pad[batch_index][c][h+2][w+j][cc] * 5.0 + data_pad[batch_index][c][h+4][w+j][cc]
            temp_expr[(1, j)] = t0 + t1
            temp_expr[(2, j)] = t0 - t1
            temp_expr[(3, j)] = t2 + t3*2.0
            temp_expr[(4, j)] = t2 - t3*2.0
            temp_expr[(5, j)] = data_pad[batch_index][c][h+1][w+j][cc] * 4.0 - data_pad[batch_index][c][h+3][w+j][cc] * 5.0 + data_pad[batch_index][c][h+5][w+j][cc]

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), compute_temp, name="temp")

    def compute_V(b, c, eps, nu, bb, cc):
        v_expr = {}
        for i in range(6):
            t0 = temp[b][c][i][4][bb][cc] - temp[b][c][i][2][bb][cc]*4.0
            t1 = temp[b][c][i][3][bb][cc] - temp[b][c][i][1][bb][cc]*4.0
            t2 = temp[b][c][i][4][bb][cc] - temp[b][c][i][2][bb][cc]
            t3 = temp[b][c][i][3][bb][cc] - temp[b][c][i][1][bb][cc]
            v_expr[(i, 0)] = temp[b][c][i][0][bb][cc] * 4.0 - temp[b][c][i][2][bb][cc] * 5.0 + temp[b][c][i][4][bb][cc]
            v_expr[(i, 1)] = t0 + t1
            v_expr[(i, 2)] = t0 - t1
            v_expr[(i, 3)] = t2 + t3*2.0
            v_expr[(i, 4)] = t2 - t3*2.0
            v_expr[(i, 5)] = temp[b][c][i][1][bb][cc] * 4.0 - temp[b][c][i][3][bb][cc] * 5.0 + temp[b][c][i][5][bb][cc]
                       
        now = tvm.const(0.0, "float32")
        for ii in range(6):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), compute_V)
    
    outs = [V]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    V = op.output(0)
    temp = s[V].op.input_tensors[0]
    data_pad = s[temp].op.input_tensors[0]
    s[data_pad].compute_inline()
    
    b, c, eps, nu, bb, cc = s[V].op.axis
    s[V].reorder(b, c, bb, eps, nu, cc)
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu]]

    fused = s[V].fuse(b, c, bb)
    s[V].parallel(fused)

    b, c, eps, nu, bb, cc = s[temp].op.axis
    s[temp].reorder(b, c, bb, eps, nu, cc)
    s[temp].vectorize(cc)
    _ = [s[temp].unroll(x) for x in [eps, nu]]
    s[temp].compute_at(s[V], fused)

    return V, s

def decl_M(data, kernel, U, V, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P //bnb, K // bna, alpha, alpha, bnb, bna), lambda b, k, eps, nu, bb, kk:
                    tvm.sum(V[b][c // bna][eps][nu][bb][c % bna] *
                            U[c // bna][k][eps][nu][c % bna][kk], axis=c), name='M')

    outs = [M]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    M = op.output(0)
    b, k, eps, nu, bb, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]

    fused = s[M].fuse(b, k)
    s[M].parallel(fused)
    co, ci = s[M].split(c, factor=8)    
    s[M].reorder(co, bb, ci, kk)
    s[M].unroll(ci)
    s[M].vectorize(kk)

    return M, s

def decl_output(data, kernel, M, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    K, C, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1

    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    def compute_temp(tile_index, k, eps, nu, kk):
        b = tile_index // bnb
        bb = tile_index % bnb
        temp_expr = {}
        for j in range(6):
            t0 =  M[b][k][1][j][bb][kk] + M[b][k][2][j][bb][kk]
            t1 =  M[b][k][3][j][bb][kk] + M[b][k][4][j][bb][kk]
            t2 =  M[b][k][1][j][bb][kk] - M[b][k][2][j][bb][kk]
            t3 =  M[b][k][3][j][bb][kk] - M[b][k][4][j][bb][kk]
            temp_expr[(0, j)] = t0 + t1 + M[b][k][0][j][bb][kk]
            temp_expr[(1, j)] = t2 + t3*2.0
            temp_expr[(2, j)] = t0 + t1*4.0
            temp_expr[(3, j)] = t2 + t3*8.0 + M[b][k][5][j][bb][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P, K // bna, m, alpha, bna), compute_temp, name="temp")

    def compute_output(n, k, h, w, kk):
        b = (n * nH * nW + (h//m) * nW + w//m)
        eps = h%m
        nu = w%m
        output_expr = {}
        for i in range(4):
            t0 =  temp[b][k][i][1][kk] + temp[b][k][i][2][kk]
            t1 =  temp[b][k][i][3][kk] + temp[b][k][i][4][kk]
            t2 =  temp[b][k][i][1][kk] - temp[b][k][i][2][kk]
            t3 =  temp[b][k][i][3][kk] - temp[b][k][i][4][kk]
            output_expr[(i, 0)] = t0 + t1 + temp[b][k][i][0][kk]
            output_expr[(i, 1)] = t2 + t3 * 2.0
            output_expr[(i, 2)] = t0 + t1 * 4.0
            output_expr[(i, 3)] = t2 + t3 * 8.0 + temp[b][k][i][5][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 output_expr[(ii, jj)],
                                 now)
        return now

    output = tvm.compute((N, K // bna, H, W, bna), compute_output)

    outs = [output]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    temp = s[output].op.input_tensors[0]

    n, k, h, w, kk = s[output].op.axis
    ho, hi = s[output].split(h, factor=4)
    wo, wi = s[output].split(w, factor=4)
    s[output].reorder(n, k, ho, wo,  hi, wi, kk)
    s[output].unroll(hi)
    s[output].unroll(wi)
    s[output].vectorize(kk)
    fused = s[output].fuse(n, k, ho, wo) 
    s[output].parallel(fused)

    b, k, eps, nu, kk = s[temp].op.axis
    s[temp].unroll(eps)
    s[temp].unroll(nu)
    s[temp].vectorize(kk)
    s[temp].compute_at(s[output], fused)
    
    return output, s

def decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m):
    def compute_temp(b, c, eps, nu, bb, cc):
        temp_expr = {}
        batch_index = (b*bnb+bb) // (nH*nW)
        h = (b*bnb+bb) // nW % nH * m
        w = (b*bnb+bb) % nW * m
        for j in range(6):
            t0 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]*4.0
            t1 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]*4.0
            t2 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]
            t3 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]
            
            temp_expr[(0, j)] = data_pad[batch_index][c][h+0][w+j][cc] * 4.0 - data_pad[batch_index][c][h+2][w+j][cc] * 5.0 + data_pad[batch_index][c][h+4][w+j][cc]
            temp_expr[(1, j)] = t0 + t1
            temp_expr[(2, j)] = t0 - t1
            temp_expr[(3, j)] = t2 + t3*2.0
            temp_expr[(4, j)] = t2 - t3*2.0
            temp_expr[(5, j)] = data_pad[batch_index][c][h+1][w+j][cc] * 4.0 - data_pad[batch_index][c][h+3][w+j][cc] * 5.0 + data_pad[batch_index][c][h+5][w+j][cc]
        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), compute_temp, name="temp")

    def compute_V(b, c, eps, nu, bb, cc):
        v_expr = {}
        for i in range(6):
            t0 = temp[b][c][i][4][bb][cc] - temp[b][c][i][2][bb][cc]*4.0
            t1 = temp[b][c][i][3][bb][cc] - temp[b][c][i][1][bb][cc]*4.0
            t2 = temp[b][c][i][4][bb][cc] - temp[b][c][i][2][bb][cc]
            t3 = temp[b][c][i][3][bb][cc] - temp[b][c][i][1][bb][cc]
            v_expr[(i, 0)] = temp[b][c][i][0][bb][cc] * 4.0 - temp[b][c][i][2][bb][cc] * 5.0 + temp[b][c][i][4][bb][cc]
            v_expr[(i, 1)] = t0 + t1
            v_expr[(i, 2)] = t0 - t1
            v_expr[(i, 3)] = t2 + t3*2.0
            v_expr[(i, 4)] = t2 - t3*2.0
            v_expr[(i, 5)] = temp[b][c][i][1][bb][cc] * 4.0 - temp[b][c][i][3][bb][cc] * 5.0 + temp[b][c][i][5][bb][cc]
                       
        now = tvm.const(0.0, "float32")
        for ii in range(6):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), compute_V)
    return V

def decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m):

    def compute_temp(tile_index, k, eps, nu, kk):
        b = tile_index // bnb
        bb = tile_index % bnb
        temp_expr = {}
        for j in range(6):
            t0 =  M[b][k][1][j][bb][kk] + M[b][k][2][j][bb][kk]
            t1 =  M[b][k][3][j][bb][kk] + M[b][k][4][j][bb][kk]
            t2 =  M[b][k][1][j][bb][kk] - M[b][k][2][j][bb][kk]
            t3 =  M[b][k][3][j][bb][kk] - M[b][k][4][j][bb][kk]
            temp_expr[(0, j)] = t0 + t1 + M[b][k][0][j][bb][kk]
            temp_expr[(1, j)] = t2 + t3*2.0
            temp_expr[(2, j)] = t0 + t1*4.0
            temp_expr[(3, j)] = t2 + t3*8.0 + M[b][k][5][j][bb][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P, K // bna, m, alpha, bna), compute_temp, name="temp")

    def compute_output(n, k, h, w, kk):
        b = (n * nH * nW + (h//m) * nW + w//m)
        eps = h%m
        nu = w%m
        output_expr = {}
        for i in range(4):
            t0 =  temp[b][k][i][1][kk] + temp[b][k][i][2][kk]
            t1 =  temp[b][k][i][3][kk] + temp[b][k][i][4][kk]
            t2 =  temp[b][k][i][1][kk] - temp[b][k][i][2][kk]
            t3 =  temp[b][k][i][3][kk] - temp[b][k][i][4][kk]
            output_expr[(i, 0)] = t0 + t1 + temp[b][k][i][0][kk]
            output_expr[(i, 1)] = t2 + t3 * 2.0
            output_expr[(i, 2)] = t0 + t1 * 4.0
            output_expr[(i, 3)] = t2 + t3 * 8.0 + temp[b][k][i][5][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 output_expr[(ii, jj)],
                                 now)
        return now

    output = tvm.compute((N, K // bna, H, W, bna), compute_output)

    return output

def decl_winograd_without_filter_transform(data, U, stride, padding, out_dtype):
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    co, ko, _, _, ci, ki  = [util.get_const_int(x) for x in U.shape]
    C = co * ci
    K = ko * ki
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")

    A_data = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 2, 4, 8],
        [1, -2, 4, -8],
        [0, 0, 0, 1]
    ], out_dtype)
    
    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    V = decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m)
    
    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P //bnb, K // bna, alpha, alpha, bnb, bna), lambda b, k, eps, nu, bb, kk:
                    tvm.sum(V[b][c // bna][eps][nu][bb][c % bna] *
                            U[c // bna][k][eps][nu][c % bna][kk], axis=c), name='M')
    
    # inverse transform
    output = decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m)
    
    return output

def schedule_winograd_without_filter_transform(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    temp_output_transform = s[output].op.input_tensors[0]
    M = s[temp_output_transform].op.input_tensors[0]
    V, U = s[M].op.input_tensors
    temp_input_transform = s[V].op.input_tensors[0]
    data_pad = s[temp_input_transform].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    # transform image
    s[data_pad].compute_inline()    
    b, c, eps, nu, bb, cc = s[V].op.axis
    s[V].reorder(b, c, bb, eps, nu, cc)
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu]]

    fused = s[V].fuse(b, c, bb)
    s[V].parallel(fused)

    b, c, eps, nu, bb, cc = s[temp_input_transform].op.axis
    s[temp_input_transform].reorder(b, c, bb, eps, nu, cc)
    s[temp_input_transform].vectorize(cc)
    _ = [s[temp_input_transform].unroll(x) for x in [eps, nu]]
    s[temp_input_transform].compute_at(s[V], fused)

    # batch gemm
    b, k, eps, nu, bb, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    fused = s[M].fuse(b, k)
    s[M].parallel(fused)
    co, ci = s[M].split(c, factor=8)    
    s[M].reorder(co, bb, ci, kk)
    s[M].unroll(ci)
    s[M].vectorize(kk)
    
    # inverse transform
    n, k, h, w, kk = s[output].op.axis
    ho, hi = s[output].split(h, factor=4)
    wo, wi = s[output].split(w, factor=4)
    s[output].reorder(n, k, ho, wo,  hi, wi, kk)
    s[output].unroll(hi)
    s[output].unroll(wi)
    s[output].vectorize(kk)
    fused = s[output].fuse(n, k, ho, wo) 
    s[output].parallel(fused)

    b, k, eps, nu, kk = s[temp_output_transform].op.axis
    s[temp_output_transform].unroll(eps)
    s[temp_output_transform].unroll(nu)
    s[temp_output_transform].vectorize(kk)
    s[temp_output_transform].compute_at(s[output], fused)

    return s

def transform_filter(w_np):
    num_filter, in_channel, kernel, kernel = w_np.shape
    G = np.array([
        [1 / 4.0, 0, 0],
        [-1 / 6.0, -1 / 6.0, -1 / 6.0],
        [-1 / 6.0, 1 / 6.0, -1 / 6.0],
        [1 / 24.0, 1 / 12.0, 1 / 6.0],
        [1 / 24.0, -1 / 12.0, 1 / 6.0],
        [0, 0, 1]
    ], dtype=np.float32)    
    bna = 8
    out = np.empty((in_channel // bna, num_filter // bna, 6, 6, bna, bna), w_np.dtype)
    for i in range(in_channel):
        for j in range(num_filter):
            out[i // bna, j // bna, :, :, i % bna, j % bna] = np.dot(G, np.dot(w_np[j, i], G.transpose()))
    return out

def nchwc_to_nchw(arr):
    n, c, h, w, cc = arr.shape
    channels = c * cc
    ret = np.zeros((n, channels, h, w))
    for i in range(channels):
        ret[:, i] = arr[:, i//cc, :, :, i%cc]
    return ret
    
def test_components(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size
    m = 4
    r = 3
    alpha = m + r - 1
    K = num_filter
    H = W = in_size
    N = batch
    C = in_channel
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    A = tvm.placeholder((batch, in_channel // bna, in_height, in_width, bna), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((C // bna, K // bna, alpha, alpha, bna, bna), name='U')
    V = tvm.placeholder((P // bnb, C // bna, alpha, alpha, bnb, bna), name='V')
    M = tvm.placeholder((P // bnb, K // bna, alpha, alpha, bnb, bna), name='M')

    output = tvm.placeholder((N, K // bna, in_size, in_size, bna), name='output')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.wino")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(nchwc_to_nchw(a_np), dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()
    u_np = transform_filter(w_np)
    v_np = np.random.uniform(size=util.get_const_tuple(V.shape)).astype(dtype)
    m_np = np.zeros(util.get_const_tuple(M.shape), dtype=dtype)
    output_np = np.zeros(util.get_const_tuple(output.shape), dtype=dtype)

    with tvm.target.create(device):
        V_out, s_V = decl_V(A, W, stride, padding, dtype)
        M_out, s_M = decl_M(A, W, U, V, stride, padding, dtype)
        output_out, s_output = decl_output(A, W, M, stride, padding, dtype)        
        
    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    v = tvm.nd.array(v_np, ctx)
    m = tvm.nd.array(m_np, ctx)
    output_tvm = tvm.nd.array(output_np, ctx)        
    num_runs = 1000
    times = {}

    with tvm.build_config(auto_unroll_max_step=500,
                          unroll_explicit=True):
        func_input_transform = tvm.build(s_V, [A, V_out], device)
        func_input_transform(a, v)
        timer = func_input_transform.time_evaluator(func_input_transform.entry_name, ctx, number=num_runs)
        times["V"] = timer(a, v).mean * 1000
        #print(tvm.lower(s_V, [A, V_out], simple_mode=True))

        func_batch_mm = tvm.build(s_M, [U, V, M_out], device)
        #print(tvm.lower(s_M, [U, V, M_out], simple_mode=True))
        func_batch_mm(u, v, m)
        
        timer = func_batch_mm.time_evaluator(func_batch_mm.entry_name, ctx, number=num_runs)
        times["M"] = timer(u, v, m).mean * 1000
        #print(tvm.lower(s_M, [A, W, U, V, M], simple_mode=True))
        func_inverse_transform = tvm.build(s_output, [M, output_out], device)
        func_inverse_transform(m, output_tvm)
        timer = func_inverse_transform.time_evaluator(func_inverse_transform.entry_name, ctx, number=num_runs)
        times["output"] = timer(m, output_tvm).mean * 1000
        #print(tvm.lower(s_output, [A, W, M, output], simple_mode=True))
        
        np.testing.assert_allclose(nchwc_to_nchw(output_tvm.asnumpy()), b_np, rtol=1e-5)
    return times

def test_winograd_without_filter_transform(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size
    bna, bnb = 8, 8

    A = tvm.placeholder((batch, in_channel // bna, in_height, in_width, bna), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((in_channel // bna, num_filter // bna, 6, 6, bna, bna), name='U')
    
    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    output = tvm.placeholder((batch, num_filter//bna, in_size, in_size, bna), name='output')
    
    @memoize("topi.tests.test_topi_conv2d_nchw.wino")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(nchwc_to_nchw(a_np), dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    with tvm.target.create(device):
        B = decl_winograd_without_filter_transform(A, U, stride, padding, dtype)
        s = schedule_winograd_without_filter_transform([B])

    u_np = transform_filter(w_np)
    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    u = tvm.nd.array(u_np, ctx)    
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    
    with tvm.build_config(auto_unroll_max_step=500,
                          unroll_explicit=True):
        func = tvm.build(s, [A, U, B], device)
        func(a, u, b)
        #print(tvm.lower(s, [A, U, B], simple_mode=True))
        num_runs = 1000
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        np.testing.assert_allclose(nchwc_to_nchw(b.asnumpy()), b_np, rtol=1e-5)
        return timer(a, u, b).mean
    

# for copy paste as markdown
def generate_table(workloads, wino_times, direct_times):
    print("| (batch,CI,size,CO) | TVM Winograd (This code) | TVM Direct |")
    print("|------------- |:-------------:|:-------------:|")
    for (workload, t_wino, t_direct) in zip(workloads, wino_times, direct_times):
        print("|", workload, "| %.3f | %.3f |" % (t_wino,  t_direct))

workloads1 = [(1, 32, 128, 16),
              (1, 16, 128, 8),
              (1, 8, 128, 16),
              (1, 16, 128, 32),
              (1, 32, 64, 32),
              (1, 32, 64, 64),
              (1, 64, 32, 64),
              (1, 64, 16, 64),
              (1, 128, 16, 64),
              (1, 128, 32, 64),
              (1, 96, 64, 32),
              (1, 40, 128, 16),
              (1, 16, 128, 16)
             ]

workloads2 = [(workload[0] * 10, *workload[1:]) for workload in workloads1] # 10 x 128 x 128
workloads3 = [(workload[0], workload[1], workload[2] * 2, workload[3]) for workload in workloads1] # 1 x 256 x 256
workloads4 = [(workload[0] * 10, workload[1], workload[2] * 2, workload[3]) for workload in workloads1] # 10 x 256 x 256

wino_times = []
direct_times = []
device = "llvm"
workloads = workloads1

for workload in workloads:
    times = test_components(*workload, 3, 1, 1, device)
    t_wino = test_winograd_without_filter_transform(*workload, 3, 1, 1, device)
    wino_times.append(t_wino * 1000)    
    t_direct = reference_direct(*workload, 3, 1, 1, device)
    direct_times.append(t_direct * 1000)
    
    print("Workload: ", workload)    
    for (k,v) in times.items():
        print("%s: %f" % (k, v))
    print("Total: %f" % np.sum(list(times.values())))
    print("Wino time: ", wino_times[-1])    
    print("Direct: %f\n" % direct_times[-1])

generate_table(workloads, wino_times, direct_times)
