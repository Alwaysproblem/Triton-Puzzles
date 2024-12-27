# %% [markdown]
# <a href="https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Triton Puzzles
#
# Programming for accelerators such as GPUs is critical for modern AI systems.
# This often means programming directly in proprietary low-level languages such as CUDA. [Triton](https://github.com/openai/triton/) is an alternative open-source language that allows you to code at a higher-level and compile to accelerators like GPU.
#
# ![image.png](./images/image1.png)
#
# Coding for Triton is very similar to Numpy and PyTorch in both syntax and semantics. However, as a lower-level language there are a lot of details that you need to keep track of. In particular, one area that learners have trouble with is memory loading and storage which is critical for speed on low-level devices.
#
# This set is puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. These puzzles **do not** need to run on GPU since they use a Triton interpreter.
#
#
#

# %%
import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32
import triton_viz
import inspect
from triton_viz.interpreter import record_builder
import triton_viz.interpreter


def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}, viz=False):
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    if viz:
        triton_viz.interpreter.record_builder.reset()
    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        print(p)
        args[n + "_ptr"] = ([d.size for d in p.annotation.dims], p)
    args["z_ptr"] = ([d.size for d in signature.return_annotation.dims], None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v).to(device="cuda") - 0.5)
        if t is not None and t.annotation.dtypes[0] == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v).to(device="cuda")
    grid = lambda meta: (
        triton.cdiv(nelem["N0"], meta["B0"]),
        triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
        triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)),
    )

    # for k, v in args.items():
    #    print(k, v)
    puzzle[grid](*tt_args, **B, **nelem)
    if viz:
        triton_viz.trace(puzzle)[grid](*tt_args, **B, **nelem)
    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    match = torch.allclose(z.cpu(), z_.cpu(), rtol=1e-3, atol=1e-3)
    print("Results match:", match)
    failures = False
    if viz:
        failures = triton_viz.launch()
    if not match or failures:
        print("Invalid Access:", failures)
        print("Yours:", z)
        print("Spec:", z_)
        print(torch.isclose(z, z_))
        return
    # PUPPIES!
    from IPython.display import HTML
    import random

    print("Correct!")
    pups = [
        "2m78jPG",
        "pn1e9TO",
        "MQCIwzT",
        "udLK6FS",
        "ZNem5o3",
        "DS2IZ6K",
        "aydRUz8",
        "MVUdQYK",
        "kLvno0p",
        "wScLiVz",
        "Z0TII8i",
        "F1SChho",
        "9hRi2jN",
        "lvzRF3W",
        "fqHxOGI",
        "1xeUYme",
        "6tVqKyM",
        "CCxZ6Wr",
        "lMW0OPQ",
        "wHVpHVG",
        "Wj2PGRl",
        "HlaTE8H",
        "k5jALH0",
        "3V37Hqr",
        "Eq2uMTA",
        "Vy9JShx",
        "g9I2ZmK",
        "Nu4RH7f",
        "sWp0Dqd",
        "bRKfspn",
        "qawCMl5",
        "2F6j2B4",
        "fiJxCVA",
        "pCAIlxD",
        "zJx2skh",
        "2Gdl1u7",
        "aJJAY4c",
        "ros6RLC",
        "DKLBJh7",
        "eyxH0Wc",
        "rJEkEw4",
    ]
    return HTML(
        """
    <video alt="test" controls autoplay=1>
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """
        % (random.sample(pups, 1)[0])
    )


# %% [markdown]
# ## Introduction
#
# To begin with, we will only use `tl.load` and `tl.store` in order to build simple programs.
#
# Here's an example of load. It takes an `arange` over the memory. By default the indexing of torch tensors with column, rows, depths or right-to-left. It also takes in a mask as the second argument. Mask is critically important because all shapes in Triton need to be powers of two.


# %%
@triton.jit
def demo(x_ptr):
    range = tl.arange(0, 8)
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, mask=(range < 5), other=0)
    print(x)


triton_viz.trace(demo)[(1, 1, 1)](torch.ones(4, 3))
# triton_viz.trace(demo)[(1, 1, 1)](torch.randint(5, size = (4, 3)))
triton_viz.launch()

# %% [markdown]
# You can also use this trick to read in a 2d array.


# %%
@triton.jit
def demo(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    print(x)


triton_viz.trace(demo)[(1, 1, 1)](torch.ones(4, 4))
triton_viz.launch()

# %% [markdown]
# The `tl.store` function is quite similar. It allows you to write to a tensor.


# %%
@triton.jit
def demo(z_ptr):
    range = tl.arange(0, 8)
    z = tl.store(z_ptr + range, 10, range < 5)


z = torch.ones(4, 3)
triton_viz.trace(demo)[(1, 1, 1)](z)
print(z)
triton_viz.launch()

# %% [markdown]
# You can only load in relatively small `blocks` at a time in Triton. to work with larger tensors you need to use a program id axis to run multiple blocks in parallel. Here is an example with one program axis with 3 blocks. You can use the visualizer to scroll over it.


# %%
@triton.jit
def demo(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 5)
    print("Print for each", pid, x)


x = torch.ones(2, 4, 4)
triton_viz.trace(demo)[(3, 1, 1)](x)
triton_viz.launch()

# %% [markdown]
# See the [Triton Docs](https://triton-lang.org/main/index.html) for further information.

# %% [markdown]
# ## Puzzle 1: Constant Add
#
# Add a constant to a vector. Uses one program id axis. Block size `B0` is always the same as vector `x` with length `N0`.
#
#
# $$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$
#

# %% [markdown]
# ![image.png](./images/image2.png)


# %%
def add_spec(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.0


@triton.jit
def add_kernel(x_ptr: torch.Tensor, z_ptr: torch.Tensor, N0, B0: tl.constexpr):
    offsets = tl.arange(0, B0)
    x = tl.load(x_ptr + offsets)
    # Finish me!
    mask = offsets < N0
    z = x + 10.0
    tl.store(z_ptr + offsets, z, mask=mask)


test(add_kernel, add_spec, nelem={"N0": 32})

# %% [markdown]
# ## Puzzle 2: Constant Add Block
#
# Add a constant to a vector. Uses one program block axis (no `for` loops yet). Block size `B0` is now smaller than the shape vector `x` which is `N0`.
#
#
# $$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$
#
#

# %% [markdown]
# ![image.png](./images/image3.png)


# %%
def add2_spec(x: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    return x + 10.0


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * B0
    offsets = block_start + tl.arange(0, B0)
    mask = offsets < N0
    x = tl.load(x_ptr + offsets, mask=mask)
    z = x + 10
    tl.store(z_ptr + offsets, z, mask=mask)


test(add_mask2_kernel, add2_spec, nelem={"N0": 200})

# %% [markdown]
# ## Puzzle 3: Outer Vector Add
#
# Add two vectors.
#
# Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
# Block size `B1` is always the same as vector `y` length `N1`.
#
#
# $$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1$$
#

# %% [markdown]
# ![image.png](./images/image4.png)


# %%
def add_vec_spec(
    x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]
) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    block_start_x = tl.program_id(axis=0)
    block_start_y = tl.program_id(axis=1)
    offset_x = tl.arange(0, B0) + block_start_x * B0
    offset_y = tl.arange(0, B1) + block_start_y * B1
    mask_x = offset_x < N0
    mask_y = offset_y < N1
    x = tl.load(x_ptr + offset_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offset_y, mask=mask_y, other=0.0)
    z = x[None, :] + y[:, None]
    # z shape is [N1, N0]
    offset_z = offset_y[:, None] * N0 + offset_x[None, :]
    tl.store(z_ptr + offset_z, z, mask=mask_x[None, :] & mask_y[:, None])


test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32})

# %% [markdown]
# ## Puzzle 4: Outer Vector Add Block
#
# Add a row vector to a column vector.
#
# Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
# Block size `B1` is always less than vector `y` length `N1`.
#
# $$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$
#

# %% [markdown]
# ![image.png]()


# %%
def add_vec_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_start_x = tl.program_id(axis=0)
    block_start_y = tl.program_id(axis=1)

    offset_x = tl.arange(0, B0) + block_start_x * B0
    offset_y = tl.arange(0, B1) + block_start_y * B1

    mask_x = offset_x < N0
    mask_y = offset_y < N1

    x = tl.load(x_ptr + offset_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offset_y, mask=mask_y, other=0.0)

    # x shape is [1, N0], y shape is [N1, 1]
    z = x[None, :] + y[:, None]

    # z shape is [N1, N0]
    offset_z = offset_x[None, :] + offset_y[:, None] * N0
    tl.store(z_ptr + offset_z, z, mask=mask_x[None, :] & mask_y[:, None])


test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90})

# triton_viz.interpreter.record_builder.reset()
# %% [markdown]
# ## Puzzle 5: Fused Outer Multiplication
#
# Multiply a row vector to a column vector and take a relu.
#
# Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
# Block size `B1` is always less than vector `y` length `N1`.
#
# $$z_{j, i} = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$
#
#

# %% [markdown]
# ![image.png](./images/image5.png)


# %%
def mul_relu_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    # 获取当前块的起始索引
    block_start_x = tl.program_id(axis=0)
    block_start_y = tl.program_id(axis=1)

    # 定义块内的偏移
    offset_x = tl.arange(0, B0) + block_start_x * B0
    offset_y = tl.arange(0, B1) + block_start_y * B1

    # 创建mask以防止访问超出范围的元素
    mask_x = offset_x < N0
    mask_y = offset_y < N1

    # 加载输入数据
    x = tl.load(x_ptr + offset_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offset_y, mask=mask_y, other=0.0)

    # 计算加和
    # x shape is [1, N0], y shape is [N1, 1]
    z = tl.where(x[None, :] * y[:, None] >= 0, x[None, :] * y[:, None], 0)

    # 计算存储位置的偏移量
    # z shape is [N1, N0]
    offset_z = offset_x[None, :] + offset_y[:, None] * N0

    # 存储结果
    tl.store(z_ptr + offset_z, z, mask=mask_x[None, :] & mask_y[:, None])


test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})

# %% [markdown]
# ## Puzzle 6: Fused Outer Multiplication - Backwards
#
#
# Backwards of a function that multiplies a matrix with a row vector and take a relu.
#
# Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
# Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz`
# is of shape `N1` by `N0`
#
# $$f(x, y) = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$
#
# $$dx_{i, j} = f_x'(x, y)_{i, j} \times dz_{i,j}$$

# %% [markdown]
# ![image.png](./images/image6.png)


# %%
def mul_relu_block_back_spec(
    x: Float32[Tensor, "90 100"],
    y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"],
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_start_x = tl.program_id(axis=0)
    block_start_y = tl.program_id(axis=1)

    offset_x = tl.arange(0, B0)
    offset_y = tl.arange(0, B1)
    index_i = block_start_x * B0 + offset_x
    index_j = block_start_y * B1 + offset_y

    mask_x = index_i < N0
    mask_y = index_j < N1
    mask = mask_x[None, :] & mask_y[:, None]

    # z shape is [N1, N0]
    offsets = index_i[None, :] + index_j[:, None] * N0

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + index_j, mask=mask_y, other=0.0)
    dz = tl.load(dz_ptr + offsets, mask=mask, other=0.0)

    z = x * y[:, None]

    relu_mask = z >= 0
    dz = tl.where(relu_mask, dz, 0.0)

    dx = dz * y[:, None]
    tl.store(dx_ptr + offsets, dx, mask=mask_x[None, :] & mask_y[:, None])

    return


test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90})

# %% [markdown]
# ## Puzzle 7: Long Sum
#
# Sum of a batch of numbers.
#
# Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
# Each element is of length `T`. Process it `B1 < T` elements at a time.
#
# $$z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0$$
#
# Hint: You will need a for loop for this problem. These work and look the same as in Python.

# %% [markdown]
# ![image.png](./images/image7.png)


# %%
def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)


@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):

    pid = tl.program_id(0)
    index_i = pid * B0 + tl.arange(0, B0)
    mask_i = index_i < N0

    z = tl.zeros([B0], dtype=tl.float32)

    for i in range(0, T, B1):
        index_j = i + tl.arange(0, B1)
        mask_j = index_j < T
        # x shape is [B0, B1]
        offset_x = index_i[:, None] * T + index_j[None, :]
        x = tl.load(x_ptr + offset_x, mask=mask_j[None, :] & mask_i[:, None], other=0.0)
        z += tl.sum(x, axis=1)

    tl.store(z_ptr + index_i, z, mask=mask_i)


test(sum_kernel, sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})

# %% [markdown]
# ## Puzzle 8: Long Softmax
#
#
# Softmax of a batch of logits.
#
# Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
# Block logit length `T`.   Process it `B1 < T` elements at a time.
#
# $$z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{ for } i = 1\ldots N_0$$
#
# Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton they recommend not using `exp` but instead using `exp2`. You need the identity
#
# $$\exp(x) = 2^{\log_2(e) x}$$
#
# Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. Hint: you will find this identity useful:
#
# $$\exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2) $$

# %% [markdown]
# ![image.png](./images/image9.png)


# %%
def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504  # Constant to convert natural log to base 2

    # Offsets for the batch dimension
    offset_n0 = pid_0 * B0 + tl.arange(0, B0)
    mask_n0 = offset_n0 < N0  # Mask to prevent out-of-bounds access

    # Initialize max values to -infinity
    max_val = tl.full([B0], -float("inf"), dtype=tl.float32)

    # First Loop: Compute the maximum value m_i for each batch
    for t in range(0, T, B1):
        # Offsets for the T dimension
        offset_t = t + tl.arange(0, B1)
        mask_t = offset_t < T  # Mask to prevent out-of-bounds access

        # Compute memory offsets for loading x
        offsets = offset_n0[:, None] * T + offset_t[None, :]  # Shape [B0, B1]
        mask = mask_n0[:, None] & mask_t[None, :]

        # Load x chunk
        x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))

        # Update max_val
        curr_max = tl.max(x, axis=1)
        max_val = tl.where(curr_max > max_val, curr_max, max_val)

    # Second Loop: Compute the sum of exponentials s_i
    sum_exp = tl.zeros([B0], dtype=tl.float32)

    for t in range(0, T, B1):
        # Offsets for the T dimension
        offset_t = t + tl.arange(0, B1)
        mask_t = offset_t < T

        # Compute memory offsets
        offsets = offset_n0[:, None] * T + offset_t[None, :]
        mask = mask_n0[:, None] & mask_t[None, :]

        # Load x chunk
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Compute x - max_val for numerical stability
        x_shifted = x - max_val[:, None]

        # Compute exponentials using exp2
        exp_x = tl.exp2(x_shifted * log2_e)

        # Accumulate the sum of exponentials
        sum_exp += tl.sum(exp_x, axis=1)

    # Third Loop: Compute the softmax probabilities and store them
    for t in range(0, T, B1):
        # Offsets for the T dimension
        offset_t = t + tl.arange(0, B1)
        mask_t = offset_t < T

        # Compute memory offsets
        offsets = offset_n0[:, None] * T + offset_t[None, :]
        mask = mask_n0[:, None] & mask_t[None, :]

        # Load x chunk
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Compute x - max_val for numerical stability
        x_shifted = x - max_val[:, None]

        # Compute exponentials
        exp_x = tl.exp2(x_shifted * log2_e)

        # Compute softmax probabilities
        softmax_x = exp_x / sum_exp[:, None]

        # Store the result
        tl.store(z_ptr + offsets, softmax_x, mask=mask)


test(
    softmax_kernel,
    softmax_spec,
    B={"B0": 1, "B1": 32},
    nelem={"N0": 4, "N1": 32, "T": 200},
)

# %% [markdown]
# ## Puzzle 9: Simple FlashAttention
#
# A scalar version of FlashAttention.
#
# Uses zero programs. Block size `B0` represents `k` of length `N0`.
# Block size `B0` represents `q` of length `N0`. Block size `B0` represents `v` of length `N0`.
# Sequence length is `T`. Process it `B1 < T` elements at a time.
#
# $$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j v_{j} \text{ for } i = 1\ldots N_0$$
#
# This can be done in 1 loop using a similar trick from the last puzzle.

# %% [markdown]
# ![image.png](./images/image10.png)


# %%
def flashatt_spec(
    q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]
) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr):
    # 获取程序的ID
    pid = tl.program_id(0)

    # 计算偏移量和掩码
    offset_i = pid * B0 + tl.arange(0, B0)
    mask_i = offset_i < N0

    # 加载 q_i
    q = tl.load(q_ptr + offset_i, mask=mask_i, other=0.0)

    # 初始化 m_i, s_i, z_i
    m_i = tl.full([B0], -float("inf"), dtype=tl.float32)
    s_i = tl.zeros([B0], dtype=tl.float32)
    z_i = tl.zeros([B0], dtype=tl.float32)

    # 遍历 T 维度
    for t in range(0, T, B0):
        offset_j = t + tl.arange(0, B0)
        mask_j = offset_j < T
        # 加载 k_j 和 v_j
        k = tl.load(k_ptr + offset_j, mask=mask_j, other=0.0)
        v = tl.load(v_ptr + offset_j, mask=mask_j, other=0.0)
        # 计算 s_ij = q_i * k_j
        s_ij = q[:, None] * k[None, :]  # 形状 [B0, B0]
        # 计算当前块的最大值
        max_s_ij = tl.max(s_ij, axis=1)
        # 更新 m_i
        new_m_i = tl.maximum(m_i, max_s_ij)
        # 计算调整因子
        exp_mi_mi = tl.exp(m_i - new_m_i)
        # 更新 s_i 和 z_i
        s_i = s_i * exp_mi_mi
        z_i = z_i * exp_mi_mi
        m_i = new_m_i
        # 计算 p = exp(s_ij - m_i)
        p = tl.exp(s_ij - m_i[:, None])
        # 累加 s_i 和 z_i
        s_i += tl.sum(p, axis=1)
        z_i += tl.sum(p * v[None, :], axis=1)

    # 计算最终结果
    result = z_i / s_i
    # 存储结果
    tl.store(z_ptr + offset_i, result, mask=mask_i)


test(flashatt_kernel, flashatt_spec, B={"B0": 32}, nelem={"N0": 200, "T": 200})

# %% [markdown]
# ## Puzzle 10: Two Dimensional Convolution
#
# A batched 2D convolution.
#
# Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
# Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.
#
# $$z_{i, j, k} = \sum_{oj, ok} k_{oj,ok} \times x_{i,j + oj, k + ok} \text{ for } i = 1\ldots N_0$$
#
#

# %% [markdown]
# ![image.png](./images/image11.png)


# %%
def conv2d_spec(
    x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]
) -> Float32[Tensor, "4 8 8"]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i : i + 4, j : j + 4]).sum(1).sum(1)
    return z


@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    pid_0 = tl.program_id(0)
    batch_offset = pid_0 * H * W

    h_range = tl.arange(0, KH)
    w_range = tl.arange(0, KW)
    k = tl.load(k_ptr + h_range[:, None] * KW + w_range[None, :])

    for row_idx in tl.range(0, H):
        for col_idx in tl.range(0, W):
            range_window = (row_idx + h_range)[:, None] * W + (col_idx + w_range)[
                None, :
            ]
            mask_window = (row_idx + h_range < H)[:, None] & (col_idx + w_range < W)[
                None, :
            ]
            x_window = tl.load(
                x_ptr + batch_offset + range_window, mask=mask_window, other=0.0
            )
            tl.store(z_ptr + batch_offset + row_idx * W + col_idx, tl.sum(x_window * k))


test(
    conv2d_kernel,
    conv2d_spec,
    B={"B0": 1},
    nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4},
)

# %% [markdown]
# ## Puzzle 11: Matrix Multiplication
#
# A blocked matrix multiplication.
#
# Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
# Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols of `y` to process out of `N1`. The middle shape is `MID`.
#
# $$z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$
#
# You are allowed to use `tl.dot` which computes a smaller mat mul.
#
# Hint: the main trick is that you can split a matmul into smaller parts.
#
# $$z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k} $$
#

# %% [markdown]
# ![image.png](./images/image12.png)


# %%
def dot_spec(
    x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]
) -> Float32[Tensor, "4 32 32"]:
    return x @ y


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    # --- 1) 获取并计算三个程序ID ---
    pid_batch = tl.program_id(0)  # 对应 batch 维度 (N2)
    pid_m = tl.program_id(1)  # 对应行 维度 (N0)
    pid_n = tl.program_id(2)  # 对应列 维度 (N1)

    # 计算批次索引 [B2]
    batch_idx = pid_batch * B2 + tl.arange(0, B2)
    mask_batch = batch_idx < N2

    # 计算行索引 [B0]
    m_idx = pid_m * B0 + tl.arange(0, B0)
    mask_m = m_idx < N0

    # 计算列索引 [B1]
    n_idx = pid_n * B1 + tl.arange(0, B1)
    mask_n = n_idx < N1

    # --- 2) 初始化累加器 z_block: 形状 [B2, B0, B1] ---
    z_block = tl.zeros((B2, B0, B1), dtype=tl.float32)

    # --- 3) 遍历 MID 维度，分块累加 ---
    # 每次处理大小为 B_MID 的子块
    for offset_mid in range(0, MID, B_MID):
        # 计算本次循环的 mid-range
        mid_range = offset_mid + tl.arange(0, B_MID)
        mask_mid = mid_range < MID

        # ------------------------
        #   加载 x_block
        # ------------------------
        # x_block形状: [B2, B0, B_MID]
        # x_offsets 计算:
        #   batch_idx[:, None, None] --> [B2, 1, 1]
        #   m_idx[None, :, None]     --> [1, B0, 1]
        #   mid_range[None, None, :] --> [1, 1, B_MID]
        # x 在内存中是按 [N2, N0, MID] 排列
        x_offsets = (
            batch_idx[:, None, None] * (N0 * MID)
            + m_idx[None, :, None] * MID
            + mid_range[None, None, :]
        )
        x_mask = (
            mask_batch[:, None, None] & mask_m[None, :, None] & mask_mid[None, None, :]
        )
        # flatten offsets & masks
        x_offsets_flat = tl.reshape(x_offsets, (B2 * B0 * B_MID,))
        x_mask_flat = tl.reshape(x_mask, (B2 * B0 * B_MID,))
        # load
        x_vals = tl.load(x_ptr + x_offsets_flat, mask=x_mask_flat, other=0.0)
        # reshape back
        x_block = tl.reshape(x_vals, (B2, B0, B_MID))

        # ------------------------
        #   加载 y_block
        # ------------------------
        # y_block形状: [B2, B_MID, B1]
        # y_offsets计算:
        #   batch_idx[:, None, None]  --> [B2, 1, 1]
        #   mid_range[:, None]        --> [B_MID, 1]
        #   n_idx[None, :]           --> [1, B1]
        # y 在内存中是按 [N2, MID, N1] 排列
        y_offsets = (
            batch_idx[:, None, None] * (MID * N1)
            + mid_range[:, None] * N1
            + n_idx[None, :]
        )
        y_mask = mask_batch[:, None, None] & mask_mid[:, None] & mask_n[None, :]
        # flatten
        y_offsets_flat = tl.reshape(y_offsets, (B2 * B_MID * B1,))
        y_mask_flat = tl.reshape(y_mask, (B2 * B_MID * B1,))
        y_vals = tl.load(y_ptr + y_offsets_flat, mask=y_mask_flat, other=0.0)
        y_block = tl.reshape(y_vals, (B2, B_MID, B1))

        # ------------------------
        #   做小块 matmul 并累加
        # ------------------------
        # x_block: [B2, B0, B_MID]
        # y_block: [B2, B_MID, B1]
        # => tl.dot => [B2, B0, B1]
        z_block += tl.dot(x_block, y_block)

    # --- 4) 将z_block存储到全局 z_ptr ---
    # z 的形状是 [N2, N0, N1]
    z_offsets = (
        batch_idx[:, None, None] * (N0 * N1)
        + m_idx[None, :, None] * N1
        + n_idx[None, None, :]
    )
    z_mask = mask_batch[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :]
    # flatten
    z_offsets_flat = tl.reshape(z_offsets, (B2 * B0 * B1,))
    z_mask_flat = tl.reshape(z_mask, (B2 * B0 * B1,))
    z_block_flat = tl.reshape(z_block, (B2 * B0 * B1,))

    # store
    tl.store(z_ptr + z_offsets_flat, z_block_flat, mask=z_mask_flat)


test(
    dot_kernel,
    dot_spec,
    B={"B0": 16, "B1": 16, "B2": 2, "B_MID": 16},
    nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32},
    # viz=True,
)


# %% [markdown]
# ## Puzzle 12: Quantized Matrix Mult
#
# When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.
#
# For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.
#
# Mathematically it looks like.
#
# $$z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$
#
# However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.
#
#
#

# %% [markdown]
# ![image.png](./images/image13.png)

# %%

FPINT = 32 // 4
GROUP = 8


def quant_dot_spec(
    scale: Float32[Tensor, "32 8"],
    offset: Int32[Tensor, "32"],
    weight: Int32[Tensor, "32 8"],
    activation: Float32[Tensor, "64 32"],
) -> Float32[Tensor, "32 32"]:
    offset = offset.view(32, 1)

    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask

    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = (
        extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    )
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation


@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)


test(
    quant_dot_kernel,
    quant_dot_spec,
    B={"B0": 16, "B1": 16, "B_MID": 64},
    nelem={"N0": 32, "N1": 32, "MID": 64},
)
