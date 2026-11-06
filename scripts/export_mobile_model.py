"""Export the actual hybrid model; reject inconsistent labels and conversion drift."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from cbam_layers import CBAM, ChannelAttention, SpatialAttention  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "mobile-assets/agrilite_hybrid.keras")
    parser.add_argument("--output", type=Path, default=ROOT / "mobile-assets/agrilite_hybrid.fp16.tflite")
    parser.add_argument("--version", default="hybrid26-qwen05-v1",
                        help="Unique release identifier; change for every new model pack")
    parser.add_argument("--catalog-output", type=Path, default=ROOT / "mobile/model-catalog.json")
    args = parser.parse_args()
    original = json.loads((ROOT / "outputs/outputs/hybrid/class_labels.json").read_text())
    display = json.loads((ROOT / "models/class_labels.json").read_text())["class_to_display"]
    model = tf.keras.models.load_model(
        args.model, compile=False,
        custom_objects={"CBAM": CBAM, "ChannelAttention": ChannelAttention,
                        "SpatialAttention": SpatialAttention},
    )
    names = original["class_names"]
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, len(names)):
        raise ValueError(f"Unsupported input/output: {model.input_shape}, {model.output_shape}")
    if any(original["class_indices"][name] != index for index, name in enumerate(names)):
        raise ValueError("Training class indices do not match label order")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    input_tensor, = interpreter.get_input_details()
    output_tensor, = interpreter.get_output_details()
    if input_tensor["dtype"] != np.float32 or output_tensor["dtype"] != np.float32:
        raise ValueError("Android runtime expects float32 input and output")
    if list(input_tensor["shape"]) != [1, 224, 224, 3] or list(output_tensor["shape"]) != [1, len(names)]:
        raise ValueError("Converted tensor shapes do not match Android contract")
    rng = np.random.default_rng(42)
    max_error = 0.0
    # These are conversion-parity probes, not an accuracy evaluation.
    probes = [np.zeros((1, 224, 224, 3), np.float32),
              np.ones((1, 224, 224, 3), np.float32)]
    probes.extend(rng.random((1, 224, 224, 3), dtype=np.float32) for _ in range(8))
    for probe in probes:
        expected = model(probe, training=False).numpy()
        interpreter.set_tensor(input_tensor["index"], probe)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_tensor["index"])
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        if not np.all(np.isfinite(actual)) or np.argmax(expected) != np.argmax(actual):
            raise ValueError("FP16 conversion changed the top class or produced invalid values")
        np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.03)
    classes = [
        {"id": name, "crop": name.split("_", 1)[0],
         "name": display.get(name, name.split("_", 1)[1].replace("_", " "))}
        for name in names
    ]
    llm = ROOT / "mobile-assets/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if not llm.exists() or llm.open("rb").read(4) != b"GGUF":
        raise ValueError("Download the GGUF LLM before generating the catalog")
    artifacts = []
    for id_, path, filename in [
        ("classifier", args.output, "agrilite_hybrid.fp16.tflite"),
        ("llm", llm, llm.name),
    ]:
        artifacts.append({"id": id_, "filename": filename,
                          "bytes": path.stat().st_size, "sha256": digest(path)})
    catalog = {
        "version": args.version, "supportedCrops": ["tomato", "chilli"],
        "classes": classes, "artifacts": artifacts,
        "classifier": {"inputShape": [1, 224, 224, 3], "normalization": "rgb/255",
                       "outputShape": [1, len(names)], "output": "probabilities"},
        "llm": {"name": "Qwen2.5-0.5B-Instruct Q4_K_M", "license": "Apache-2.0",
                "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "revision": "9217f5db79a29953eb74d5343926648285ec7e67"},
    }
    args.catalog_output.write_text(json.dumps(catalog, indent=2) + "\n")
    report = {"sourceSha256": digest(args.model), "convertedSha256": digest(args.output),
              "parityProbes": len(probes), "maxAbsoluteError": max_error,
              "fieldAccuracyEvaluated": False}
    (args.output.parent / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
