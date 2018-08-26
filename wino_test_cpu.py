import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi import util
from topi.nn import pad

bna = 8
bnb = 8
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
    # with tvm.build_config(auto_unroll_max_step=500,
    #                       unroll_explicit=True):
    func = tvm.build(s1, [A, W, B], device)
    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
    num_runs = 1000
    timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    return timer(a, w, b).mean

def reference_direct_NCHWc(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size
    ic_block = 8
    oc_block = 8
    A = tvm.placeholder((batch, in_channel//ic_block, in_height, in_width, ic_block), name='A')
    W = tvm.placeholder((num_filter//oc_block, in_channel//ic_block, kernel, kernel, ic_block, oc_block), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.reference_direct")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(nchwc_to_nchw(a_np), nchwc_to_nchw_kernel(w_np), stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    with tvm.target.create(device):
        B = topi.nn.conv2d_NCHWc(A, W, num_filter=num_filter, kernel_size=(3,3), stride=1, padding=1, layout='NCHWc', out_layout='NCHWc', out_dtype='float32')
        s1 = topi.generic.schedule_conv2d_NCHWc(num_filter=num_filter, kernel_size=(3,3), strides=1, padding=1, layout='NCHWc', out_layout='NCHWc', outs=[B])
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    # with tvm.build_config(auto_unroll_max_step=500,
    #                       unroll_explicit=True):
    func = tvm.build(s1, [A, W, B], device)
    func(a, w, b)
    np.testing.assert_allclose(nchwc_to_nchw(b.asnumpy()), b_np, rtol=1e-5)
    num_runs = 1000
    timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    return timer(a, w, b).mean
    
def decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m):
    def compute_temp(b, c, eps, nu, cc):
        temp_expr = {}
        batch_index = b // (nH*nW)
        h = b // nW % nH * m
        w = b % nW * m
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

    temp = tvm.compute((P, C // bna, alpha, alpha, bna), compute_temp, name="temp_V")

    def compute_V(b, c, eps, nu, cc):
        v_expr = {}
        for i in range(6):
            t0 = temp[b][c][i][4][cc] - temp[b][c][i][2][cc]*4.0
            t1 = temp[b][c][i][3][cc] - temp[b][c][i][1][cc]*4.0
            t2 = temp[b][c][i][4][cc] - temp[b][c][i][2][cc]
            t3 = temp[b][c][i][3][cc] - temp[b][c][i][1][cc]
            v_expr[(i, 0)] = temp[b][c][i][0][cc] * 4.0 - temp[b][c][i][2][cc] * 5.0 + temp[b][c][i][4][cc]
            v_expr[(i, 1)] = t0 + t1
            v_expr[(i, 2)] = t0 - t1
            v_expr[(i, 3)] = t2 + t3*2.0
            v_expr[(i, 4)] = t2 - t3*2.0
            v_expr[(i, 5)] = temp[b][c][i][1][cc] * 4.0 - temp[b][c][i][3][cc] * 5.0 + temp[b][c][i][5][cc]
                       
        now = tvm.const(0.0, "float32")
        for ii in range(6):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((P, C // bna, alpha, alpha, bna), compute_V, name="V")
    return V

def decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m):

    def compute_temp(b, k, eps, nu, kk):
        temp_expr = {}
        for j in range(6):
            t0 =  M[b][k][1][j][kk] + M[b][k][2][j][kk]
            t1 =  M[b][k][3][j][kk] + M[b][k][4][j][kk]
            t2 =  M[b][k][1][j][kk] - M[b][k][2][j][kk]
            t3 =  M[b][k][3][j][kk] - M[b][k][4][j][kk]
            temp_expr[(0, j)] = t0 + t1 + M[b][k][0][j][kk]
            temp_expr[(1, j)] = t2 + t3*2.0
            temp_expr[(2, j)] = t0 + t1*4.0
            temp_expr[(3, j)] = t2 + t3*8.0 + M[b][k][5][j][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P, K // bna, m, alpha, bna), compute_temp, name="temp_Y")

    def compute_output(b, k, eps, nu, kk):
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

    Y = tvm.compute((P, K // bna, m, m, bna), compute_output, name="Y")
    output = tvm.compute((N, K // bna, H, W, bna), lambda n, k, h, w, kk:
                         Y[n * nH * nW + (h//m) * nW + w//m][k][h % m][w % m][kk],
                         name='output', tag='winograd_conv_output')
    
    return output

def decl_winograd_without_filter_transform(data, U, stride, padding, out_dtype):
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    ko, _, _, C, ki  = [util.get_const_int(x) for x in U.shape]
    C = co * ci
    K = ko * ki
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")

    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    V = decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m)
    
    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P, K // bna, alpha, alpha, bna), lambda b, k, eps, nu, kk:
                    tvm.sum(V[b][c // bna][eps][nu][c % bna] *
                            U[k][eps][nu][c][kk], axis=c), name='M')
    
    # inverse transform
    output = decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m)
    
    return output

def schedule_winograd_without_filter_transform(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    Y = s[output].op.input_tensors[0]
    temp_output_transform = s[Y].op.input_tensors[0]
    M = s[temp_output_transform].op.input_tensors[0]
    V, U = s[M].op.input_tensors
    temp_input_transform = s[V].op.input_tensors[0]
    data_pad = s[temp_input_transform].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    b_factor = 8
    P = V.shape[0].value
    if P == 16:
        b_factor = 2

    # transform image
    s[data_pad].compute_inline()    
    b, c, eps, nu, cc = s[V].op.axis
    bo, bi = s[V].split(b, factor=b_factor)
    s[V].reorder(bo, c, bi, eps, nu, cc)    
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu]]

    b, c, eps, nu, cc = s[temp_input_transform].op.axis
    s[temp_input_transform].vectorize(cc)
    _ = [s[temp_input_transform].unroll(x) for x in [eps, nu]]
    s[temp_input_transform].compute_at(s[V], bi)

    # batch gemm
    b, k, eps, nu, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    co, ci = s[M].split(c, factor=8)
    bo, bi = s[M].split(b, factor=b_factor)
    s[M].reorder(bo, k, bi, eps, co, nu, ci, kk)
    s[V].compute_at(s[M], bo)
    s[M].vectorize(kk)
    
    # inverse transform
    b, k, eps, nu, kk = s[Y].op.axis
    bo, bi = s[Y].split(b, factor=b_factor)
    s[Y].reorder(bo, k, bi, eps, nu, kk)
    #s[Y].parallel(bo)
    s[Y].vectorize(kk)
    _ = [s[Y].unroll(x) for x in [eps, nu]]
    #s[M].compute_at(s[Y], bo)
    
    b, k, eps, nu, kk = s[temp_output_transform].op.axis
    s[temp_output_transform].unroll(eps)
    s[temp_output_transform].unroll(nu)
    s[temp_output_transform].vectorize(kk)
    s[temp_output_transform].compute_at(s[Y], bi)
    
    n, k, h, w, kk = s[output].op.axis
    ho, hi = s[output].split(h, factor=4)
    wo, wi = s[output].split(w, factor=4)
    s[output].reorder(n, ho, wo, k, hi, wi, kk)
    woo, bi = s[output].split(wo, factor=b_factor)    
    bo = s[output].fuse(n, ho, woo)
    s[output].reorder(bo, k, bi, hi, wi, kk)    
    s[output].vectorize(kk)
    s[output].parallel(bo)
    s[M].compute_at(s[output], bo)
    s[Y].compute_at(s[output], bo)

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
    out = np.empty((num_filter // bna, 6, 6, in_channel, bna), w_np.dtype)
    for i in range(in_channel):
        for j in range(num_filter):
            out[j // bna, :, :, i, j % bna] = np.dot(G, np.dot(w_np[j, i], G.transpose()))
    return out

def nchwc_to_nchw(arr):
    n, c, h, w, cc = arr.shape
    channels = c * cc
    ret = np.zeros((n, channels, h, w))
    for i in range(channels):
        ret[:, i] = arr[:, i//cc, :, :, i%cc]
    return ret

def nchwc_to_nchw_kernel(kernel):
    n, c, h, w, ic, oc  = kernel.shape
    in_channels = c * ic
    out_channels = n * oc
    ret = np.zeros((out_channels, in_channels, h, w))
    for i in range(out_channels):
        for j in range(in_channels):
            ret[i, j] = kernel[i//oc, j//ic, :, :, j%ic, i%oc]
    return ret


def test_winograd_without_filter_transform(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel // bna, in_height, in_width, bna), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((num_filter // bna, 6, 6, in_channel, bna), name='U')
    
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
    
    with tvm.build_config(auto_unroll_max_step=100,
                          unroll_explicit=True):
        func = tvm.build(s, [A, U, B], device)
        func(a, u, b)
        #print(tvm.lower(s, [A, U, B], simple_mode=True))
        # with open("wino.s", "w") as fo:
        #     fo.write(func.get_source("asm"))
        num_runs = 1000
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        np.testing.assert_allclose(nchwc_to_nchw(b.asnumpy()), b_np, rtol=1e-5)
        return timer(a, u, b).mean


# for copy paste as markdown
def generate_table(workloads, wino_times, direct_times, nchwc_times):
    print("| (batch,CI,size,CO) | Winograd | NCHW | NCHWc")
    print("|------------- |:-------------:|:-------------:|")
    for (workload, t_wino, t_direct, t_nchwc) in zip(workloads, wino_times, direct_times, nchwc_times):
        print("|", workload, "| %.3f | %.3f | %.3f | " % (t_wino,  t_direct, t_nchwc))

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

vgg_workloads = [(1, 64, 224, 64), #relu, input and output transform slow
                 (1, 64, 112, 128),#relu2
                 (1, 128, 112, 128),
                 (1, 128, 56, 256),
                 (1, 256, 56, 256), #relu4
                 (1, 256, 28, 512),
                 (1, 512, 28, 512), # relu6
                 (1, 512, 14, 512)] # relu7

workloads2 = [(workload[0] * 10, *workload[1:]) for workload in workloads1] # 10 x 128 x 128
workloads3 = [(workload[0], workload[1], workload[2] * 2, workload[3]) for workload in workloads1] # 1 x 256 x 256
workloads4 = [(workload[0] * 10, workload[1], workload[2] * 2, workload[3]) for workload in workloads1] # 10 x 256 x 256


wino_times = []
direct_times = []
nchwc_times = []
device = "llvm -mcpu=core-avx2"
workloads = workloads3

for workload in workloads:
    t_nchwc = reference_direct_NCHWc(*workload, 3, 1, 1, device)
    nchwc_times.append(t_nchwc * 1000)

    t_wino = test_winograd_without_filter_transform(*workload, 3, 1, 1, device)
    wino_times.append(t_wino * 1000)

    t_direct = reference_direct(*workload, 3, 1, 1, device)
    direct_times.append(t_direct * 1000)
    
    print("Wino time: ", wino_times[-1])
    print("NCHWc time: ", nchwc_times[-1])

generate_table(workloads, wino_times, direct_times, nchwc_times)
