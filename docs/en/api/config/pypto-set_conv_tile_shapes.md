# pypto.set_conv_tile_shapes

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:34.933Z pushedAt=2026-08-26T09:10:38.130Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the **TileShape** size of each dimension under the L1/L0 cache hierarchy in convolution (conv) computation, and controls the enable state of the **L0TileInfo** configuration switch.

## Prototype

```python
def set_conv_tile_shapes(tile_l1_info: pypto_impl.TileL1Info, tile_l0_info: pypto_impl.TileL0Info = None) -> None
```

## Parameters

| Parameter    | Input/Output | Description                                   |
|--------------|--------------|-----------------------------------------------|
| **tile_l1_info** | Input        | **TileShape** configuration information for convolution computation at the L1 cache hierarchy. |
| **tile_l0_info** | Input        | **TileShape** configuration information for convolution computation at the L0 cache hierarchy. |

## Return Value

void

## Constraints

**TileShape** must satisfy the following constraints:

**Note: Both L1 and L0 have tileN, but they represent different meanings. tileL1Info.tileN represents the number of output channels, while tileL0Info.tileN represents the size of n at the L0 level.**

- Alignment constraints:

    - The values of each dimension of **tileL1Info** must satisfy the range constraints:

        - 1 <= tileHin <= Hin (where Hin is the actual height of the input feature map)

        - When wout % 16 = 0, 1 <= tileHout <= Hout (where Hout is the actual height of the output feature map)

        - When wout % 16 != 0, tileHout = 1

        - 1 <= tileWin <= Win (where Win is the actual width of the input feature map)

        - 1 <= tileWout <= CeilAlign(Wout, 16) (where Wout is the actual width of the output feature map). tileWout must satisfy 16-element alignment, that is, `tileWout % 16 == 0`

        - 1 <= tileCinFmap <= Cin (Cin is the actual number of channels of the input feature map).

        - tileCinFmap * sizeof(dtype) % 32 == 0

        - 1 <= tileCinWeight <= Cin (Cin is the actual number of input channels of the weight).

        - tileCinWeight * sizeof(dtype) % 32 == 0

        - 1 <= tileN <= CeilAlign(Cout // groups, 16) (Cout is the actual number of channels of the output feature map).

        - tileN % 16 == 0

        - **tileBatch** = 1 (represents the batch number)

    - The values of each dimension in **tileL0Info** must satisfy the alignment constraints:

        - **tileK** must satisfy: `C0 <= tileK <= min(kAL1, kBL1)`

        - **tileK** must satisfy: `tileK % C0 == 0`

        - **tileK** `kAL1 % tilek == 0`

        - **tilek** `kBL1 % tilek == 0`

        - **tileW** must satisfy 16-element alignment, that is, `tileW % 16 == 0`

        - **tileW** must satisfy: `1 <= tileW <= tileWout`

        - **tileH** must satisfy: `1 <= tileH <= tileHout`

        - tileN (representing the size of n at the L0 level) must satisfy 16-element alignment, that is, `tileN % 16 == 0`.

        - tileN must satisfy: `1 <= tileN <= CeilAlign(tileL1Info.tileN, 16)`.

        where:

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

    - L0 and L1 dimension hierarchy constraints:

        - 1 <= tileL0Info.tileH <= tileL1Info.tileHout and tileL1Info.tileHout % tileL0Info.tileH == 0

        - 1 <= tileL0Info.tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileL0Info.tileW == 0

        - 1 <= tileL0Info.tileN <= tileL1Info.tileN

- Buffer spatial constraints:

    - L0A, L0B, and L0C spatial constraints:

        ```txt
        CeilAlign(tileH * tileW, 16)* CeilAlign(tileK, C0) * sizeof(dtype) <= L0A_size

        CeilAlign(tileK, C0) * CeilAlign(tileN, 16) * sizeof(dtype) <= L0B_size

        CeilAlign(tileH * tileW, 16)* CeilAlign(tileN, 16) * sizeof(FP32) <= L0C_size
        ```

        Where:

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `L0A_size = 65536 bytes`

        - `L0B_size = 65536 bytes`

        - `L0C_size = 131072 bytes`

        - `ALIGN_SIZE_32 = 32`

    - L1 spatial constraint:

        ```txt
        CeilAlign(hinL1 * winL1 * kAL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(nL1 * kBL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(tileN * sizeof(dtype), ALIGN_SIZE_32) <= L1_size
        ```

        Where:

        - `hinL1 = min((tileHout - 1) * strideH + (Kh - 1) * dilationH + 1, Hin)` (where Hin is the input feature map height)

        - `winL1 = min((tileWout - 1) * strideW + (Kw - 1) * dilationW + 1, Win)` (where Win is the input feature map width)

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `nL1 = tileN` (number of output channels)

        - `dtype is the data type of input_conv (input matrix).`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

        - `CeilAlign(value, align) {  return ((value + align - 1) // align) * align;}`

- Special scenario constraints:

    - When `tileL0Info` is not passed in, the default `TileL0Info` instance is automatically used, and the L0TileInfo switch is automatically disabled. When a valid `tileL0Info` is passed in, the L0TileInfo switch is automatically enabled.

    - The convolution kernel/channel dimension configuration must match the number of input and output channels and the kernel size of the actual convolution operator to prevent the tile size from exceeding the operator dimension range.

## Example

```python
# Construct the L1 tile configuration (ensure all values are within the valid range).
l1_tile = pypto_impl.TileL1Info(
    tileHin=4,        # Must satisfy 1 <= tileHin <= Hin.
    tileHout=4,       # Must satisfy 1 <= tileHout <= Hout.
    tileWin=8,        # Must satisfy 1 <= tileWin <= Win.
    tileWout=8,       # Must satisfy 1 <= tileWout <= Wout.
    tileCinFmap=16,   # Must satisfy 1 <= tileCinFmap <= Cin.
    tileCinWeight=32, # Must satisfy 1 <= tileCinWeight <= Cin.
    tileN=16,         # Must satisfy 1 <= tileN <= Cout.
    tileBatch=1       # Must satisfy tileBatch = 1.
)

# Construct the L0 tile configuration (satisfying the alignment constraints).
l0_tile = pypto_impl.TileL0Info(
    tileH=2,   # Must satisfy tileH <= tileL1Info.tileHout and tileL1Info.tileHout % tileH == 0.
    tileW=8,   # Must satisfy tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileW == 0.
    tileK=32,  # Must satisfy tileK * sizeof(dtype) % 32 == 0 (assuming dtype is FP16 and sizeof = 2, then 32 * 2 = 64, and 64 % 32 = 0).
    tileN=16   # Must satisfy tileN % 16 == 0.
)

# Set the convolution TileShape (enable L0TileInfo).
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile, tile_l0_info=l0_tile)

# Set only the L1 TileShape (disable L0TileInfo).
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile)
```
