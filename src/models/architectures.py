"""Task 2.2 & 2.3: Model architectures for VisionGuard (23L0713_A2_Ops).

Includes:
  - Baseline CNN built from scratch (Conv2D, BatchNormalization, MaxPooling2D, Dropout, Sigmoid head).
  - Pretrained Transfer Learning models (MobileNetV2, EfficientNetB0, ResNet50) with two-stage
    freeze/unfreeze support and backbone-specific input preprocessing.
"""

import io
from pathlib import Path

import pandas as pd
import tensorflow as tf


def get_metrics() -> list:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_baseline_cnn(
    img_size: tuple[int, int] = (160, 160),
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.35,
) -> tf.keras.Model:
    """Task 2.2: Custom Baseline CNN built from scratch."""
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="input_rgb")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.75, name="drop2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop3")(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_fc")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_fc")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_fc")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_Baseline_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics(),
    )
    return model


def build_transfer_model(
    backbone_name: str = "MobileNetV2",
    img_size: tuple[int, int] = (160, 160),
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.35,
    unfreeze_last_n: int = 0,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Task 2.3: Pretrained Transfer Learning model (Stage 1 frozen or Stage 2 fine-tuned)."""
    input_shape = (img_size[0], img_size[1], 3)
    inputs = tf.keras.Input(shape=input_shape, name="input_rgb")

    if backbone_name == "MobileNetV2":
        # MobileNetV2 expects [-1, 1] scaling: (x / 127.5) - 1.0 (matches mobilenet_v2.preprocess_input)
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "EfficientNetB0":
        # EfficientNetB0 expects [0, 255] float32 directly inside its built-in Rescaling layer
        x = inputs
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "ResNet50":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="resnet50_scale")(inputs)
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = False
    if unfreeze_last_n > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_last_n:]:
            # Keep BatchNormalization in inference stats mode during fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.7, name="head_drop2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"VisionGuard_{backbone_name}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics(),
    )
    return model, base_model


def unfreeze_last_block(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_last_n: int = 20,
    fine_tune_lr: float = 5e-5,
) -> list[str]:
    """Unfreeze the last `unfreeze_last_n` layers of `base_model` (excluding BatchNorm) and recompile."""
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_last_n:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics(),
    )
    trainable_names = [
        layer.name for layer in base_model.layers if layer.trainable and layer.count_params() > 0
    ] + [
        layer.name
        for layer in model.layers
        if layer is not base_model and layer.trainable and layer.count_params() > 0
    ]
    return trainable_names


def export_model_summary_and_table(
    model: tf.keras.Model,
    summary_txt_path: Path,
    table_csv_path: Path,
) -> pd.DataFrame:
    """Save Keras model.summary() text and a structured per-layer shape & parameter table."""
    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    summary_str = buf.getvalue()
    summary_txt_path.write_text(summary_str, encoding="utf-8")

    rows = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = "Multiple/Nested"
        params = int(layer.count_params())
        trainable_params = int(sum( tf.keras.backend.count_params(w) for w in layer.trainable_weights ))
        non_trainable_params = params - trainable_params
        rows.append(
            {
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": out_shape,
                "Total Params": params,
                "Trainable Params": trainable_params,
                "Non-Trainable Params": non_trainable_params,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(table_csv_path, index=False)
    return df
