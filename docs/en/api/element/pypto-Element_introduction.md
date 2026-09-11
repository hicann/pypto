# Introduction to **pypto.Element**

<!-- md-trans-meta sourceCommit=0c9d4972ac0ba19cdc885dac450d7436b8603e8f translatedAt=2026-08-20T10:23:06.290Z pushedAt=2026-08-20T13:02:21.133Z -->

In the PyPTO framework, the **Element** type is used to store scalar constants for constant expressions in computation operations. The built-in Python **int** type is typically mapped to the **DT_INT64** type, while the **float** type is mapped to the **DT_FP32** type. In scenarios where both **Tensor** and **Scalar** operand types are present, **int** and **float** are usually converted to the data type of the corresponding tensor.
