"""Explicit metadata for every condition class the trained checkpoints know about.

Each condition maps to (crop, is_healthy). This table is written out by hand from prepare_data.PV_MAP / PD_MAP (the folder
-> label mapping used to build the training data), so health and crop membership never depend on parsing class names
at inference time. validate() checks the table against a checkpoint's label map and reports any class that is missing
from either side, so a future label map with new classes fails loudly instead of being silently mis-assigned.
"""

CONDITION_META = {
    "Apple___Apple_scab": ("Apple", False),
    "Apple___Black_rot": ("Apple", False),
    "Apple___Cedar_apple_rust": ("Apple", False),
    "Apple___healthy": ("Apple", True),
    "Cherry___Powdery_mildew": ("Cherry", False),
    "Cherry___healthy": ("Cherry", True),
    "Grape___Black_rot": ("Grape", False),
    "Grape___Esca": ("Grape", False),
    "Grape___Leaf_blight": ("Grape", False),
    "Grape___healthy": ("Grape", True),
    "Orange___Haunglongbing": ("Orange", False),
    "Peach___Bacterial_spot": ("Peach", False),
    "Peach___healthy": ("Peach", True),
    "Pepper_bell___Bacterial_spot": ("Pepper_bell", False),
    "Pepper_bell___healthy": ("Pepper_bell", True),
    "Strawberry___Leaf_scorch": ("Strawberry", False),
    "Strawberry___healthy": ("Strawberry", True),
    "Tomato___Bacterial_spot": ("Tomato", False),
    "Tomato___Early_blight": ("Tomato", False),
    "Tomato___Late_blight": ("Tomato", False),
    "Tomato___Leaf_Mold": ("Tomato", False),
    "Tomato___Mosaic_virus": ("Tomato", False),
    "Tomato___Septoria_leaf_spot": ("Tomato", False),
    "Tomato___Spider_mites": ("Tomato", False),
    "Tomato___Target_Spot": ("Tomato", False),
    "Tomato___Yellow_Leaf_Curl_Virus": ("Tomato", False),
    "Tomato___healthy": ("Tomato", True),
}


def validate(conditions, crops, meta=CONDITION_META):
    """Return a list of problems (empty = label map and metadata agree).

    conditions: the checkpoint's condition list; crops: the checkpoint's crop list.
    """
    problems = []
    unknown = [c for c in conditions if c not in meta]
    if unknown:
        problems.append(f"condition classes without metadata: {unknown}")
    missing = [c for c in meta if c not in conditions]
    if missing:
        problems.append(f"metadata classes absent from the label map: {missing}")
    bad_crop = sorted({meta[c][0] for c in conditions if c in meta} - set(crops))
    if bad_crop:
        problems.append(f"metadata crops not in the crop list: {bad_crop}")
    return problems
