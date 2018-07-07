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

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")    
    
    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), lambda b, c, eps, nu, bb, cc:
                    tvm.sum(data_pad[(b*bnb+bb) // (nH*nW)][c][(b*bnb+bb) // nW % nH * m + r_eps][(b*bnb+bb) % nW * m + r_nu][cc] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')
    
    outs = [V]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    V = op.output(0)
    data_pad, B = s[V].op.input_tensors
    s[data_pad].compute_inline()
    s[B].compute_inline()
    b, c, eps, nu, bb, cc = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_nu, r_eps, bb, cc)
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)

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

    m = 2
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

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    # inverse transform
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K // bna, H, W, bna), lambda n, k, h, w, kk: 
                    tvm.sum(M[(n * nH * nW + (h//m) * nW + w//m)//bna][k][r_eps][r_nu][(n * nH * nW + (h//m) * nW + w//m)%bna][kk] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')

    outs = [output]
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    _, A = s[output].op.input_tensors
    s[A].compute_inline()

    n, k, h, w, kk = s[output].op.axis
    r_eps, r_nu = s[output].op.reduce_axis    
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(n, k, ho, wo, hi, wi, r_eps, r_nu, kk)
    s[output].vectorize(kk)
    fused = s[output].fuse(n, k, ho, wo)
    s[output].parallel(fused)
    
    return output, s
    
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
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), lambda b, c, eps, nu, bb, cc:
                    tvm.sum(data_pad[(b*bnb+bb) // (nH*nW)][c][(b*bnb+bb) // nW % nH * m + r_eps][(b*bnb+bb) % nW * m + r_nu][cc] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')
    
    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P //bnb, K // bna, alpha, alpha, bnb, bna), lambda b, k, eps, nu, bb, kk:
                    tvm.sum(V[b][c // bna][eps][nu][bb][c % bna] *
                            U[c // bna][k][eps][nu][c % bna][kk], axis=c), name='M')
    
    # inverse transform
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K // bna, H, W, bna), lambda n, k, h, w, kk: 
                    tvm.sum(M[(n * nH * nW + (h//m) * nW + w//m)//bna][k][r_eps][r_nu][(n * nH * nW + (h//m) * nW + w//m)%bna][kk] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')
    
    return output

def schedule_winograd_without_filter_transform(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    M, A = s[output].op.input_tensors
    V, U = s[M].op.input_tensors
    data_pad, B = s[V].op.input_tensors
    data = s[data_pad].op.input_tensors[0]

    # transform image
    s[data_pad].compute_inline()    
    s[B].compute_inline()
    b, c, eps, nu, bb, cc = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_nu, r_eps, bb, cc)
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)

    # batch gemm
    b, k, eps, nu, bb, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    fused = s[M].fuse(b, k)
    s[M].parallel(fused)
    co, ci = s[M].split(c, factor=8)    
    s[M].reorder(co, bb, ci, kk)
    s[M].unroll(ci)
    s[M].vectorize(kk)
    
#     # inverse transform
    s[A].compute_inline()
    n, k, h, w, kk = s[output].op.axis
    r_eps, r_nu = s[output].op.reduce_axis    
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(n, k, ho, wo, hi, wi, r_eps, r_nu, kk)
    s[output].vectorize(kk)
    fused = s[output].fuse(n, k, ho, wo)
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
    bna = 8
    out = np.empty((in_channel // bna, num_filter // bna, 4, 4, bna, bna), w_np.dtype)
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
    u_np = np.random.uniform(size=util.get_const_tuple(U.shape)).astype(dtype)
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
    num_runs = 100
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
    U = tvm.placeholder((in_channel // bna, num_filter // bna, 4, 4, bna, bna), name='U')
    
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
        #print(tvm.lower(s, [A, W, B], simple_mode=True))
        num_runs = 100
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
              (1, 64, 8, 64),
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
