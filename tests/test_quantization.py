"""Tests for app/quantization/ - packing/unpacking, BitLinear, save/load utilities.

All tests run on CPU without a GPU.
"""

import torch


# ---------------------------------------------------------------------------
# Packing / Unpacking (packing.py)
# ---------------------------------------------------------------------------

class TestPackTernaryWeights:
    def test_round_trip_basic(self):
        """pack + unpack produces original tensor."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = torch.tensor([-1, 0, 1, -1, 0, 1, 1, 0], dtype=torch.int8)
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_round_trip_with_padding(self):
        """Pack/unpack works when length is not a multiple of 4."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = torch.tensor([1, -1, 0, 1, -1], dtype=torch.int8)  # length 5
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_all_zeros(self):
        """Handles all-zero tensor."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = torch.zeros(16, dtype=torch.int8)
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_all_ones(self):
        """Handles all-ones tensor."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = torch.ones(16, dtype=torch.int8)
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_all_negative_ones(self):
        """Handles all negative ones."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = -torch.ones(16, dtype=torch.int8)
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_2d_tensor(self):
        """Works with 2D weight tensor."""
        from app.quantization.packing import pack_ternary_weights, unpack_ternary_weights

        w = torch.randint(-1, 2, (8, 8), dtype=torch.int8)
        packed, shape = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, shape)
        assert torch.equal(w, unpacked)

    def test_compression_ratio(self):
        """Packed tensor is ~4x smaller."""
        from app.quantization.packing import pack_ternary_weights

        w = torch.randint(-1, 2, (256,), dtype=torch.int8)
        packed, _ = pack_ternary_weights(w)
        # 256 elements -> 64 bytes packed
        assert packed.numel() == 64

    def test_verify_packing_correctness(self):
        """verify_packing_correctness returns True for valid data."""
        from app.quantization.packing import verify_packing_correctness

        w = torch.randint(-1, 2, (32,), dtype=torch.int8)
        assert verify_packing_correctness(w) is True


class TestPackForTriton:
    def test_round_trip(self):
        """Triton-format pack + unpack produces original 2D tensor."""
        from app.quantization.packing import pack_for_triton, unpack_for_triton

        w = torch.randint(-1, 2, (16, 32), dtype=torch.int8)
        packed, shape = pack_for_triton(w)
        unpacked = unpack_for_triton(packed, shape)
        assert torch.equal(w, unpacked)

    def test_with_padding(self):
        """Triton-format handles non-multiple-of-4 in_features."""
        from app.quantization.packing import pack_for_triton, unpack_for_triton

        w = torch.randint(-1, 2, (8, 13), dtype=torch.int8)
        packed, shape = pack_for_triton(w)
        unpacked = unpack_for_triton(packed, shape)
        assert torch.equal(w, unpacked)

    def test_packed_shape(self):
        """Packed tensor has correct shape [out_features, in_features // 4]."""
        from app.quantization.packing import pack_for_triton

        w = torch.randint(-1, 2, (16, 32), dtype=torch.int8)
        packed, _ = pack_for_triton(w)
        assert packed.shape == (16, 8)

    def test_verify_triton_packing_correctness(self):
        """verify_triton_packing_correctness returns True for valid data."""
        from app.quantization.packing import verify_triton_packing_correctness

        w = torch.randint(-1, 2, (16, 32), dtype=torch.int8)
        assert verify_triton_packing_correctness(w) is True


class TestGetPackedSize:
    def test_known_shapes(self):
        from app.quantization.packing import get_packed_size

        assert get_packed_size((8,)) == 2
        assert get_packed_size((16,)) == 4
        assert get_packed_size((4, 4)) == 4
        assert get_packed_size((5,)) == 2  # ceil(5/4) = 2

    def test_large_shape(self):
        from app.quantization.packing import get_packed_size

        assert get_packed_size((1024, 1024)) == 262144


# ---------------------------------------------------------------------------
# BitLinear (bitlinear.py)
# ---------------------------------------------------------------------------

class TestBitLinear:
    def test_forward_shape(self):
        """BitLinear forward produces correct output shape."""
        from app.quantization.bitlinear import BitLinear

        layer = BitLinear(in_features=32, out_features=16, bias=True, dtype=torch.float32)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_forward_no_bias(self):
        """BitLinear works without bias."""
        from app.quantization.bitlinear import BitLinear

        layer = BitLinear(in_features=16, out_features=8, bias=False, dtype=torch.float32)
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_from_linear(self):
        """from_linear creates a quantized layer from nn.Linear."""
        from app.quantization.bitlinear import BitLinear
        import torch.nn as nn

        linear = nn.Linear(64, 32)
        bit = BitLinear.from_linear(linear)

        assert bit.in_features == 64
        assert bit.out_features == 32
        assert bit.weight_ternary.dtype == torch.int8
        # Check values are in {-1, 0, 1}
        unique = bit.weight_ternary.unique().tolist()
        for val in unique:
            assert val in [-1, 0, 1]

    def test_from_linear_preserves_bias(self):
        """from_linear copies bias from original layer."""
        from app.quantization.bitlinear import BitLinear
        import torch.nn as nn

        linear = nn.Linear(16, 8, bias=True)
        linear.bias.data.fill_(1.5)
        bit = BitLinear.from_linear(linear)
        assert bit.bias is not None
        assert torch.allclose(bit.bias, linear.bias)

    def test_from_linear_no_bias(self):
        """from_linear works with no-bias linear."""
        from app.quantization.bitlinear import BitLinear
        import torch.nn as nn

        linear = nn.Linear(16, 8, bias=False)
        bit = BitLinear.from_linear(linear)
        assert bit.bias is None

    def test_weight_scale_is_positive(self):
        """Weight scale should be positive after from_linear."""
        from app.quantization.bitlinear import BitLinear
        import torch.nn as nn

        linear = nn.Linear(64, 32)
        bit = BitLinear.from_linear(linear)
        assert bit.weight_scale.item() > 0

    def test_extra_repr(self):
        """extra_repr includes quantized label."""
        from app.quantization.bitlinear import BitLinear

        layer = BitLinear(16, 8, bias=True, dtype=torch.float32)
        assert "quantized=1.58-bit" in layer.extra_repr()

    def test_get_memory_savings(self):
        """Memory savings calculation is reasonable."""
        from app.quantization.bitlinear import BitLinear

        layer = BitLinear(256, 128, bias=True, dtype=torch.float32)
        savings = layer.get_memory_savings()
        assert savings["original_bytes"] > savings["quantized_bytes"]
        assert savings["savings_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Save / Load utilities (utils.py)
# ---------------------------------------------------------------------------

class TestSaveLoadQuantized:
    def test_save_creates_files(self, tmp_path):
        """save_quantized_model creates config.json and weights file."""
        from app.quantization.utils import save_quantized_model
        import torch.nn as nn

        model = nn.Linear(16, 8)
        output_dir = str(tmp_path / "quantized")

        save_quantized_model(
            unet=model,
            output_dir=output_dir,
            base_model_id="test/model",
            quantized_layers=["weight"],
        )

        assert (tmp_path / "quantized" / "unet_quantized.pt").exists()
        assert (tmp_path / "quantized" / "config.json").exists()

    def test_save_config_contents(self, tmp_path):
        """Saved config.json has expected fields."""
        import json
        from app.quantization.utils import save_quantized_model
        import torch.nn as nn

        model = nn.Linear(16, 8)
        output_dir = str(tmp_path / "quantized")

        save_quantized_model(
            unet=model,
            output_dir=output_dir,
            base_model_id="test/model",
            quantized_layers=["layer1", "layer2"],
            metadata={"extra": "info"},
        )

        with open(tmp_path / "quantized" / "config.json") as f:
            cfg = json.load(f)

        assert cfg["base_model_id"] == "test/model"
        assert cfg["num_quantized_layers"] == 2
        assert cfg["metadata"]["extra"] == "info"


class TestGetModelSize:
    def test_model_size_calculation(self):
        """get_model_size_mb returns reasonable size."""
        from app.quantization.utils import get_model_size_mb
        import torch.nn as nn

        model = nn.Linear(1024, 1024, bias=False)
        size_mb = get_model_size_mb(model)
        # 1024 * 1024 * 4 bytes (float32) = 4 MB
        assert 3.9 < size_mb < 4.1
