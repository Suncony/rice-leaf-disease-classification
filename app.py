from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List

from PIL import Image, ImageTk, UnidentifiedImageError
import torch
from torch import nn
from torchvision.models import efficientnet_b0

from prepare_data import EVAL_TRANSFORM

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "trained_model.pth"
SUPPORTED_ARCHITECTURE = "efficientnet_b0"
SUPPORTED_WEIGHTS = "EfficientNet_B0_Weights.DEFAULT"
IMAGE_PREVIEW_SIZE = (320, 320)
TOP_K_PREDICTIONS = 3
WINDOW_TITLE = "Rice Leaf Disease Classifier"


class RiceLeafClassifierApp:
    # Initialize the UI state, runtime model state, and top-level window configuration.
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.minsize(760, 520)

        self.device = self.get_device()
        self.checkpoint_path = DEFAULT_CHECKPOINT_PATH
        self.model: nn.Module | None = None
        self.transform = None
        self.class_names: List[str] = []
        self.selected_image_path: Path | None = None
        self.selected_image: Image.Image | None = None
        self.preview_image: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="Upload an image to begin.")
        self.image_name_var = tk.StringVar(value="No image selected.")
        self.prediction_vars = [tk.StringVar(value=f"Top {index + 1}: -") for index in range(TOP_K_PREDICTIONS)]

        self.build_ui()

    # Choose GPU when available so inference uses the best device on the current machine.
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Build the Tkinter layout for image preview, prediction labels, and action buttons.
    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.grid(sticky="nsew")
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(header_frame, text=WINDOW_TITLE, font=("Segoe UI", 15, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header_frame,
            text="Upload a rice leaf image and run inference to view the top predictions.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding=14)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        preview_inner = ttk.Frame(preview_frame, padding=10)
        preview_inner.grid(row=0, column=0, sticky="nsew")
        preview_inner.columnconfigure(0, weight=1)
        preview_inner.rowconfigure(0, weight=1)

        self.preview_box = ttk.Frame(preview_inner, width=IMAGE_PREVIEW_SIZE[0], height=IMAGE_PREVIEW_SIZE[1])
        self.preview_box.grid(row=0, column=0)
        self.preview_box.grid_propagate(False)

        self.preview_label = ttk.Label(
            self.preview_box,
            text="No image uploaded.",
            anchor="center",
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        details_frame = ttk.LabelFrame(main_frame, text="Details", padding=14)
        details_frame.grid(row=1, column=1, sticky="nsew")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(3, weight=1)

        image_info_frame = ttk.Frame(details_frame)
        image_info_frame.grid(row=0, column=0, sticky="ew")
        image_info_frame.columnconfigure(0, weight=1)

        ttk.Label(image_info_frame, text="Selected File", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(image_info_frame, textvariable=self.image_name_var, wraplength=260).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

        button_frame = ttk.Frame(details_frame)
        button_frame.grid(row=1, column=0, sticky="ew", pady=(16, 16))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        upload_button = ttk.Button(button_frame, text="Upload Image", command=self.on_upload)
        upload_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        predict_button = ttk.Button(button_frame, text="Run Prediction", command=self.on_predict)
        predict_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        predictions_frame = ttk.LabelFrame(details_frame, text="Top Predictions", padding=12)
        predictions_frame.grid(row=2, column=0, sticky="nsew")
        predictions_frame.columnconfigure(0, weight=1)

        for index, prediction_var in enumerate(self.prediction_vars, start=1):
            ttk.Label(
                predictions_frame,
                textvariable=prediction_var,
                font=("Segoe UI", 11, "bold" if index == 1 else "normal"),
            ).grid(row=index - 1, column=0, sticky="w", pady=4)

        status_frame = ttk.LabelFrame(details_frame, text="Status", padding=12)
        status_frame.grid(row=4, column=0, sticky="ew", pady=(16, 0))
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(status_frame, textvariable=self.status_var, wraplength=260, foreground="#1f3a5f").grid(
            row=0, column=0, sticky="w"
        )

    # Let the user pick an image file, validate it, and refresh the preview area.
    def on_upload(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected_file:
            return

        image_path = Path(selected_file)
        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            self.set_error_state(f"Could not open image: {error}")
            return

        self.selected_image_path = image_path
        self.selected_image = image
        self.image_name_var.set(f"Selected: {image_path.name}")
        self.reset_prediction_labels()
        self.update_preview(image)
        self.status_var.set("Image uploaded. Click Predict to run inference.")

    # Run inference for the current uploaded image and push the ranked results into the UI.
    def on_predict(self) -> None:
        if self.selected_image is None:
            self.set_error_state("Upload an image first.")
            return

        try:
            self.ensure_model_loaded()
            predictions = self.predict_image(self.selected_image)
        except ValueError as error:
            self.set_error_state(str(error))
            return
        except RuntimeError as error:
            self.set_error_state(f"Inference failed: {error}")
            return

        for index, (class_name, confidence) in enumerate(predictions):
            self.prediction_vars[index].set(f"Top {index + 1}: {class_name} ({confidence:.2f}%)")

        self.status_var.set("Prediction complete.")

    # Load and cache the checkpoint, class labels, transform, and model the first time prediction runs.
    def ensure_model_loaded(self) -> None:
        if self.model is not None and self.transform is not None and self.class_names:
            return

        checkpoint = self.load_checkpoint(self.checkpoint_path)
        self.class_names = self.extract_class_names(checkpoint)
        self.transform = self.resolve_transform()
        self.model = self.build_model(checkpoint, num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    # Read the trained checkpoint from disk and validate that the core payload structure exists.
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, object]:
        if not checkpoint_path.exists():
            raise ValueError(f"Missing trained checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint payload is invalid: expected a metadata dictionary.")
        self.validate_checkpoint_metadata(checkpoint)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing 'model_state_dict'.")
        return checkpoint

    # Reject checkpoints with missing or incompatible metadata before model reconstruction starts.
    def validate_checkpoint_metadata(self, checkpoint: Dict[str, object]) -> None:
        architecture = checkpoint.get("architecture")
        if architecture != SUPPORTED_ARCHITECTURE:
            raise ValueError(
                f"Unsupported checkpoint architecture '{architecture}'. Expected '{SUPPORTED_ARCHITECTURE}'."
            )

        weights = checkpoint.get("weights")
        if weights != SUPPORTED_WEIGHTS:
            raise ValueError(f"Unsupported checkpoint weights '{weights}'. Expected '{SUPPORTED_WEIGHTS}'.")

    # Convert the saved class-to-index mapping into ordered display labels for prediction output.
    def extract_class_names(self, checkpoint: Dict[str, object]) -> List[str]:
        class_to_idx = checkpoint.get("class_to_idx")
        if not isinstance(class_to_idx, dict) or not class_to_idx:
            raise ValueError("Checkpoint is missing a valid 'class_to_idx' mapping.")

        try:
            ordered_names = sorted(class_to_idx.items(), key=lambda item: int(item[1]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Checkpoint class mapping is invalid: {error}") from error

        return [class_name for class_name, _ in ordered_names]

    # Import the shared evaluation transform so uploaded images use the same preprocessing as testing.
    def resolve_transform(self):
        return EVAL_TRANSFORM

    # Rebuild the EfficientNet-B0 classifier head so the saved weights can be loaded for inference.
    def build_model(self, checkpoint: Dict[str, object], num_classes: int) -> nn.Module:
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    # Preprocess the uploaded image, run the model once, and return the top ranked classes with confidences.
    def predict_image(self, image: Image.Image) -> List[tuple[str, float]]:
        if self.model is None or self.transform is None or not self.class_names:
            raise ValueError("Model is not ready for inference.")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        top_k = min(TOP_K_PREDICTIONS, len(self.class_names))
        confidence_scores, class_indices = torch.topk(probabilities, k=top_k)

        predictions: List[tuple[str, float]] = []
        for score, class_index in zip(confidence_scores.tolist(), class_indices.tolist()):
            predictions.append((self.class_names[class_index], score * 100.0))
        return predictions

    # Resize the selected image for the window and replace the preview widget contents.
    def update_preview(self, image: Image.Image) -> None:
        preview_image = image.copy()
        preview_image.thumbnail(IMAGE_PREVIEW_SIZE)
        photo_image = ImageTk.PhotoImage(preview_image)
        self.preview_image = photo_image
        self.preview_label.configure(image=photo_image, text="")

    # Reset the prediction labels so a newly uploaded image does not show stale inference results.
    def reset_prediction_labels(self) -> None:
        for index, prediction_var in enumerate(self.prediction_vars, start=1):
            prediction_var.set(f"Top {index}: -")

    # Surface an error to both the status text and a modal dialog without closing the app.
    def set_error_state(self, message: str) -> None:
        self.status_var.set(message)
        messagebox.showerror("Prediction Error", message)


# Launch the Tkinter application and start the desktop event loop.
def main() -> None:
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    RiceLeafClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
