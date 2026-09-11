# pypto.set\_semantic\_label

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:08:09.035Z pushedAt=2026-08-26T09:10:38.134Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets a semantic label for a code segment. PyPTO Toolkit identifies the code lines following the label until the next semantic label is encountered, which helps users locate issues more conveniently.

## Prototype

```python
set_semantic_label(label: str) -> None
```

## Parameters

| Parameter  | Input/Output | Description                  |
|---------|-----------|-----------------------|
| **label**   | Input      | Name of the semantic label. Any string is supported. |

## Return Value

No return value. The setting takes effect immediately upon success.

## Constraints

None

## Example

```python
pypto.set_semantic_label("kv")
compressed_kv = pypto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
...
```
