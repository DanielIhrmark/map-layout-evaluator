import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gspread
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REVIEWERS = ["Henrik", "Daniel", "Thomas", "Ahmad", "Jonas"]

RATINGS_SHEET_NAME = "Sheet1"
HUMAN_ANNOTATIONS_SHEET_NAME = "HumanAnnotations"
GEMINI_PREDICTIONS_SHEET_NAME = "GeminiPredictions"
EVALUATION_RESULTS_SHEET_NAME = "EvaluationResults"

DISPLAY_WIDTH = 1200

SUPPORTED_IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/jpg",
    "image/png",
]


# ---------------------------------------------------------------------
# Google clients
# ---------------------------------------------------------------------

@st.cache_resource
def get_google_clients():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
    )

    gspread_client = gspread.authorize(creds)
    drive_service = build("drive", "v3", credentials=creds)

    return gspread_client, drive_service


@st.cache_resource
def get_spreadsheet(sheet_url: str):
    gspread_client, _ = get_google_clients()
    return gspread_client.open_by_url(sheet_url)


@st.cache_resource
def get_worksheet(sheet_url: str, worksheet_name: str):
    spreadsheet = get_spreadsheet(sheet_url)
    return spreadsheet.worksheet(worksheet_name)


def get_or_create_worksheet(
    sheet_url: str,
    worksheet_name: str,
    headers: List[str],
    rows: int = 2000,
):
    spreadsheet = get_spreadsheet(sheet_url)

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=rows,
            cols=max(len(headers), 10),
        )
        worksheet.update("A1", [headers])
        return worksheet

    existing_header = worksheet.row_values(1)

    if existing_header != headers:
        worksheet.update("A1", [headers])

    return worksheet


# ---------------------------------------------------------------------
# Drive handling
# ---------------------------------------------------------------------

@st.cache_data(ttl=1800)
def list_maps_in_folder(folder_id: str) -> List[Dict[str, str]]:
    _, drive_service = get_google_clients()

    files = []
    page_token = None

    mime_query = " or ".join(
        [f"mimeType = '{mime_type}'" for mime_type in SUPPORTED_IMAGE_MIME_TYPES]
    )

    while True:
        response = (
            drive_service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents and trashed = false and "
                    f"({mime_query})"
                ),
                fields="nextPageToken, files(id, name, mimeType)",
                orderBy="name",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        files.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


@st.cache_data(ttl=1800)
def download_drive_image(file_id: str) -> bytes:
    _, drive_service = get_google_clients()

    request = drive_service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return fh.getvalue()


def prepare_display_image(
    image_bytes: bytes,
    display_width: int = DISPLAY_WIDTH,
) -> Tuple[Image.Image, Image.Image, float]:
    image_stream = io.BytesIO(image_bytes)

    original_image = Image.open(image_stream)
    original_image = original_image.convert("RGB")

    original_width, original_height = original_image.size

    if original_width <= display_width:
        display_image = original_image.copy()
        scale = 1.0
    else:
        scale = display_width / original_width
        display_height = int(original_height * scale)
        display_image = original_image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )

    display_image = display_image.convert("RGB")

    return original_image, display_image, scale


# ---------------------------------------------------------------------
# Rating sheet handling
# ---------------------------------------------------------------------

def ensure_ratings_headers(sheet_url: str):
    worksheet = get_worksheet(sheet_url, RATINGS_SHEET_NAME)
    expected_header = ["Map"] + REVIEWERS
    header = worksheet.row_values(1)

    if header != expected_header:
        worksheet.update("A1", [expected_header])
        load_ratings_df.clear()


@st.cache_data(ttl=1800)
def load_ratings_df(sheet_url: str) -> pd.DataFrame:
    worksheet = get_worksheet(sheet_url, RATINGS_SHEET_NAME)
    records = worksheet.get_all_records()

    if not records:
        df = pd.DataFrame(columns=["Map"] + REVIEWERS)
    else:
        df = pd.DataFrame(records)

    for col in ["Map"] + REVIEWERS:
        if col not in df.columns:
            df[col] = ""

    df = df[["Map"] + REVIEWERS].fillna("")
    return df


def normalize_rating(value: Any) -> str:
    return str(value).strip().lower()


def add_rating_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for reviewer in REVIEWERS:
        df[reviewer] = df[reviewer].apply(normalize_rating)

    def count_rating(row, rating):
        return sum(row[reviewer] == rating for reviewer in REVIEWERS)

    df["n_easy"] = df.apply(lambda row: count_rating(row, "easy"), axis=1)
    df["n_medium"] = df.apply(lambda row: count_rating(row, "medium"), axis=1)
    df["n_difficult"] = df.apply(lambda row: count_rating(row, "difficult"), axis=1)
    df["n_irrelevant"] = df.apply(lambda row: count_rating(row, "irrelevant"), axis=1)

    df["n_ratings"] = df[REVIEWERS].apply(
        lambda row: sum(str(value).strip() != "" for value in row),
        axis=1,
    )

    df["has_disagreement"] = df[["n_easy", "n_medium", "n_difficult"]].apply(
        lambda row: sum(value > 0 for value in row) > 1,
        axis=1,
    )

    return df


def filter_maps_by_ratings(
    files: List[Dict[str, str]],
    ratings_df: pd.DataFrame,
    rating_filter: str,
    reviewer_filter: str,
    minimum_number_of_ratings: int,
    exclude_irrelevant: bool,
    only_disagreement: bool,
) -> List[Dict[str, str]]:
    available_names = {file["name"] for file in files}

    df = add_rating_summary_columns(ratings_df)
    df = df[df["Map"].isin(available_names)]

    if exclude_irrelevant:
        df = df[df["n_irrelevant"] == 0]

    if minimum_number_of_ratings > 0:
        df = df[df["n_ratings"] >= minimum_number_of_ratings]

    if only_disagreement:
        df = df[df["has_disagreement"]]

    if rating_filter != "Any rating":
        if reviewer_filter == "Any reviewer":
            reviewer_match = df[REVIEWERS].eq(rating_filter).any(axis=1)
        else:
            reviewer_match = df[reviewer_filter].eq(rating_filter)

        df = df[reviewer_match]

    selected_names = set(df["Map"].tolist())

    return [file for file in files if file["name"] in selected_names]


# ---------------------------------------------------------------------
# Sheet schemas
# ---------------------------------------------------------------------

def human_annotation_headers() -> List[str]:
    return [
        "timestamp",
        "map",
        "file_id",
        "annotator",
        "image_width",
        "image_height",
        "annotation_json",
    ]


def gemini_prediction_headers() -> List[str]:
    return [
        "timestamp",
        "map",
        "file_id",
        "model",
        "prompt_version",
        "image_width",
        "image_height",
        "prediction_json",
        "raw_response",
    ]


def evaluation_result_headers() -> List[str]:
    return [
        "timestamp",
        "map",
        "file_id",
        "annotator",
        "model",
        "prompt_version",
        "iou_threshold",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "matches_json",
    ]


# ---------------------------------------------------------------------
# Human annotation handling
# ---------------------------------------------------------------------

def reset_annotation_state_for_new_map(file_id: str):
    if "active_file_id" not in st.session_state:
        st.session_state.active_file_id = file_id

    if "current_box_start" not in st.session_state:
        st.session_state.current_box_start = None

    if "human_boxes_display" not in st.session_state:
        st.session_state.human_boxes_display = []

    if "last_processed_click" not in st.session_state:
        st.session_state.last_processed_click = None

    if st.session_state.active_file_id != file_id:
        st.session_state.active_file_id = file_id
        st.session_state.current_box_start = None
        st.session_state.human_boxes_display = []
        st.session_state.last_processed_click = None


def make_annotation_preview(
    display_image: Image.Image,
    boxes_display: List[Dict[str, int]],
    current_box_start: Optional[Tuple[int, int]],
) -> Image.Image:
    preview = display_image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes_display, start=1):
        draw.rectangle(
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
            outline="red",
            width=3,
        )
        draw.text(
            (box["x_min"] + 3, box["y_min"] + 3),
            f"h_{i:03d}",
            fill="red",
            font=font,
        )

    if current_box_start is not None:
        x_start, y_start = current_box_start
        draw.ellipse(
            [x_start - 5, y_start - 5, x_start + 5, y_start + 5],
            fill="blue",
        )
        draw.text(
            (x_start + 8, y_start + 8),
            "start",
            fill="blue",
            font=font,
        )

    return preview


def display_boxes_to_original_boxes(
    boxes_display: List[Dict[str, int]],
    scale: float,
) -> List[Dict[str, Any]]:
    boxes = []

    for i, box in enumerate(boxes_display, start=1):
        boxes.append(
            {
                "id": f"h_{i:03d}",
                "class": "text_area",
                "x_min": round(box["x_min"] / scale, 2),
                "y_min": round(box["y_min"] / scale, 2),
                "x_max": round(box["x_max"] / scale, 2),
                "y_max": round(box["y_max"] / scale, 2),
            }
        )

    return boxes


def save_human_annotation(
    sheet_url: str,
    map_name: str,
    file_id: str,
    annotator: str,
    image_width: int,
    image_height: int,
    boxes: List[Dict[str, Any]],
):
    worksheet = get_or_create_worksheet(
        sheet_url,
        HUMAN_ANNOTATIONS_SHEET_NAME,
        human_annotation_headers(),
    )

    worksheet.append_row(
        [
            datetime.now().isoformat(timespec="seconds"),
            map_name,
            file_id,
            annotator,
            image_width,
            image_height,
            json.dumps(boxes, ensure_ascii=False),
        ]
    )

    load_human_annotations_df.clear()


@st.cache_data(ttl=300)
def load_human_annotations_df(sheet_url: str) -> pd.DataFrame:
    try:
        worksheet = get_worksheet(sheet_url, HUMAN_ANNOTATIONS_SHEET_NAME)
    except gspread.WorksheetNotFound:
        return pd.DataFrame(columns=human_annotation_headers())

    records = worksheet.get_all_records()

    if not records:
        return pd.DataFrame(columns=human_annotation_headers())

    df = pd.DataFrame(records)

    for col in human_annotation_headers():
        if col not in df.columns:
            df[col] = ""

    return df[human_annotation_headers()].fillna("")


def get_latest_human_annotation(
    sheet_url: str,
    map_name: str,
    annotator: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = load_human_annotations_df(sheet_url)

    if df.empty:
        return None

    df = df[df["map"] == map_name]

    if annotator:
        df = df[df["annotator"] == annotator]

    if df.empty:
        return None

    latest = df.iloc[-1].to_dict()

    try:
        latest["boxes"] = json.loads(latest["annotation_json"])
    except json.JSONDecodeError:
        latest["boxes"] = []

    return latest


# ---------------------------------------------------------------------
# Gemini handling
# ---------------------------------------------------------------------

def get_gemini_client():
    api_key = st.secrets["gemini"]["api_key"]
    return genai.Client(api_key=api_key)


def build_gemini_prompt(
    image_width: int,
    image_height: int,
    prompt_version: str,
) -> str:
    return f"""
You are analyzing a historical map image.

Task:
Identify all visible text-bearing regions on the map.

Return only valid JSON in this exact structure:

{{
  "regions": [
    {{
      "id": "g_001",
      "class": "text_area",
      "x_min": 0,
      "y_min": 0,
      "x_max": 0,
      "y_max": 0,
      "snippet": "short readable text snippet",
      "confidence": "low"
    }}
  ]
}}

Coordinate rules:
- Use pixel coordinates relative to the original image.
- The original image width is {image_width} pixels.
- The original image height is {image_height} pixels.
- x_min and x_max must be between 0 and {image_width}.
- y_min and y_max must be between 0 and {image_height}.
- x_min must be smaller than x_max.
- y_min must be smaller than y_max.

What to include:
- Place names.
- Title text.
- Legends.
- Cartouches.
- Marginal text.
- Labels.
- Handwritten annotations.
- Any other visible text-bearing regions.

What not to include:
- Decorative borders without text.
- Empty map areas.
- Pure symbols without text.

Snippet rule:
- For each region, include a short snippet of the visible text if readable.
- If the text is visible but not readable, use an empty string.

Confidence rule:
- Use one of: "low", "medium", "high".

Prompt version: {prompt_version}
""".strip()


def clean_gemini_json_text(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def validate_and_normalize_gemini_regions(
    parsed: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    raw_regions = parsed.get("regions", [])

    if not isinstance(raw_regions, list):
        return []

    normalized_regions = []

    for i, region in enumerate(raw_regions, start=1):
        try:
            x_min = float(region.get("x_min", 0))
            y_min = float(region.get("y_min", 0))
            x_max = float(region.get("x_max", 0))
            y_max = float(region.get("y_max", 0))
        except (TypeError, ValueError):
            continue

        x_min = max(0, min(image_width, x_min))
        x_max = max(0, min(image_width, x_max))
        y_min = max(0, min(image_height, y_min))
        y_max = max(0, min(image_height, y_max))

        if x_max <= x_min or y_max <= y_min:
            continue

        confidence = str(region.get("confidence", "low")).strip().lower()

        if confidence not in ["low", "medium", "high"]:
            confidence = "low"

        normalized_regions.append(
            {
                "id": str(region.get("id", f"g_{i:03d}")),
                "class": "text_area",
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2),
                "snippet": str(region.get("snippet", "")),
                "confidence": confidence,
            }
        )

    return normalized_regions


def run_gemini_text_area_detection(
    image_bytes: bytes,
    mime_type: str,
    image_width: int,
    image_height: int,
    model_name: str,
    prompt_version: str,
) -> Tuple[List[Dict[str, Any]], str]:
    client = get_gemini_client()

    prompt = build_gemini_prompt(
        image_width=image_width,
        image_height=image_height,
        prompt_version=prompt_version,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text or ""
    cleaned_text = clean_gemini_json_text(raw_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return [], raw_text

    regions = validate_and_normalize_gemini_regions(
        parsed=parsed,
        image_width=image_width,
        image_height=image_height,
    )

    return regions, raw_text


def save_gemini_prediction(
    sheet_url: str,
    map_name: str,
    file_id: str,
    model_name: str,
    prompt_version: str,
    image_width: int,
    image_height: int,
    regions: List[Dict[str, Any]],
    raw_response: str,
):
    worksheet = get_or_create_worksheet(
        sheet_url,
        GEMINI_PREDICTIONS_SHEET_NAME,
        gemini_prediction_headers(),
    )

    worksheet.append_row(
        [
            datetime.now().isoformat(timespec="seconds"),
            map_name,
            file_id,
            model_name,
            prompt_version,
            image_width,
            image_height,
            json.dumps(regions, ensure_ascii=False),
            raw_response,
        ]
    )

    load_gemini_predictions_df.clear()


@st.cache_data(ttl=300)
def load_gemini_predictions_df(sheet_url: str) -> pd.DataFrame:
    try:
        worksheet = get_worksheet(sheet_url, GEMINI_PREDICTIONS_SHEET_NAME)
    except gspread.WorksheetNotFound:
        return pd.DataFrame(columns=gemini_prediction_headers())

    records = worksheet.get_all_records()

    if not records:
        return pd.DataFrame(columns=gemini_prediction_headers())

    df = pd.DataFrame(records)

    for col in gemini_prediction_headers():
        if col not in df.columns:
            df[col] = ""

    return df[gemini_prediction_headers()].fillna("")


def get_latest_gemini_prediction(
    sheet_url: str,
    map_name: str,
    model_name: Optional[str] = None,
    prompt_version: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = load_gemini_predictions_df(sheet_url)

    if df.empty:
        return None

    df = df[df["map"] == map_name]

    if model_name:
        df = df[df["model"] == model_name]

    if prompt_version:
        df = df[df["prompt_version"] == prompt_version]

    if df.empty:
        return None

    latest = df.iloc[-1].to_dict()

    try:
        latest["regions"] = json.loads(latest["prediction_json"])
    except json.JSONDecodeError:
        latest["regions"] = []

    return latest


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def box_area(box: Dict[str, Any]) -> float:
    width = max(0, float(box["x_max"]) - float(box["x_min"]))
    height = max(0, float(box["y_max"]) - float(box["y_min"]))
    return width * height


def calculate_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    x_left = max(float(box_a["x_min"]), float(box_b["x_min"]))
    y_top = max(float(box_a["y_min"]), float(box_b["y_min"]))
    x_right = min(float(box_a["x_max"]), float(box_b["x_max"]))
    y_bottom = min(float(box_a["y_max"]), float(box_b["y_max"]))

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    union_area = box_area(box_a) + box_area(box_b) - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def evaluate_detections(
    human_boxes: List[Dict[str, Any]],
    predicted_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    candidates = []

    for human_index, human_box in enumerate(human_boxes):
        for predicted_index, predicted_box in enumerate(predicted_boxes):
            candidates.append(
                {
                    "human_index": human_index,
                    "predicted_index": predicted_index,
                    "human_id": human_box.get("id", f"h_{human_index}"),
                    "predicted_id": predicted_box.get("id", f"p_{predicted_index}"),
                    "iou": calculate_iou(human_box, predicted_box),
                }
            )

    candidates = sorted(candidates, key=lambda item: item["iou"], reverse=True)

    used_human = set()
    used_predicted = set()
    matches = []

    for candidate in candidates:
        if candidate["iou"] < iou_threshold:
            continue

        human_index = candidate["human_index"]
        predicted_index = candidate["predicted_index"]

        if human_index in used_human or predicted_index in used_predicted:
            continue

        used_human.add(human_index)
        used_predicted.add(predicted_index)
        matches.append(candidate)

    true_positives = len(matches)
    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(human_boxes) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        sum(match["iou"] for match in matches) / len(matches)
        if matches
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "matches": matches,
    }


def save_evaluation_result(
    sheet_url: str,
    map_name: str,
    file_id: str,
    annotator: str,
    model_name: str,
    prompt_version: str,
    iou_threshold: float,
    result: Dict[str, Any],
):
    worksheet = get_or_create_worksheet(
        sheet_url,
        EVALUATION_RESULTS_SHEET_NAME,
        evaluation_result_headers(),
    )

    worksheet.append_row(
        [
            datetime.now().isoformat(timespec="seconds"),
            map_name,
            file_id,
            annotator,
            model_name,
            prompt_version,
            iou_threshold,
            result["true_positives"],
            result["false_positives"],
            result["false_negatives"],
            round(result["precision"], 4),
            round(result["recall"], 4),
            round(result["f1"], 4),
            round(result["mean_iou"], 4),
            json.dumps(result["matches"], ensure_ascii=False),
        ]
    )


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def draw_boxes_on_image(
    image: Image.Image,
    human_boxes: Optional[List[Dict[str, Any]]] = None,
    gemini_boxes: Optional[List[Dict[str, Any]]] = None,
    scale: float = 1.0,
) -> Image.Image:
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)

    human_boxes = human_boxes or []
    gemini_boxes = gemini_boxes or []

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in human_boxes:
        x_min = float(box["x_min"]) * scale
        y_min = float(box["y_min"]) * scale
        x_max = float(box["x_max"]) * scale
        y_max = float(box["y_max"]) * scale

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=3,
        )
        draw.text(
            (x_min + 3, y_min + 3),
            f"H: {box.get('id', '')}",
            fill="red",
            font=font,
        )

    for box in gemini_boxes:
        x_min = float(box["x_min"]) * scale
        y_min = float(box["y_min"]) * scale
        x_max = float(box["x_max"]) * scale
        y_max = float(box["y_max"]) * scale

        label = f"G: {box.get('id', '')}"
        snippet = str(box.get("snippet", "")).strip()

        if snippet:
            label += f" | {snippet[:25]}"

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="blue",
            width=3,
        )
        draw.text(
            (x_min + 3, y_min + 15),
            label,
            fill="blue",
            font=font,
        )

    return rendered


def make_metrics_df(result: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "true_positives": result["true_positives"],
                "false_positives": result["false_positives"],
                "false_negatives": result["false_negatives"],
                "precision": round(result["precision"], 4),
                "recall": round(result["recall"], 4),
                "f1": round(result["f1"], 4),
                "mean_iou": round(result["mean_iou"], 4),
            }
        ]
    )


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Map Text Area Detection Evaluation",
    layout="wide",
)

st.title("Map Text Area Detection Evaluation")
st.write(
    "Select maps from the existing Drive folder using the difficulty ratings sheet, "
    "mark human text-area annotations, run Gemini, and compare the results."
)

sheet_url = st.secrets["app"]["sheet_url"]
folder_id = st.secrets["app"]["drive_folder_id"]

with st.sidebar.expander("Maintenance"):
    if st.button("Verify ratings sheet headers"):
        try:
            ensure_ratings_headers(sheet_url)
            st.success("Ratings sheet headers verified.")
        except Exception as e:
            st.error("Could not verify ratings sheet headers.")
            st.exception(e)

    if st.button("Refresh ratings cache"):
        load_ratings_df.clear()
        st.success("Ratings cache cleared. Reload the app to fetch fresh ratings.")

    if st.button("Refresh Drive file cache"):
        list_maps_in_folder.clear()
        download_drive_image.clear()
        st.success("Drive cache cleared. Reload the app to fetch fresh files.")

default_gemini_model = st.secrets.get("gemini", {}).get(
    "model",
    "gemini-3-pro-preview",
)

try:
    files = list_maps_in_folder(folder_id)
    ratings_df = load_ratings_df(sheet_url)
except Exception as e:
    st.error("Could not load maps or ratings from Google services.")
    st.exception(e)
    st.stop()

if not files:
    st.warning("No image files were found in the Drive folder.")
    st.stop()


# ---------------------------------------------------------------------
# Sidebar selection
# ---------------------------------------------------------------------

st.sidebar.header("Selection")

annotator = st.sidebar.selectbox(
    "Annotator",
    REVIEWERS,
    index=None,
    placeholder="Select your name",
)

rating_filter = st.sidebar.selectbox(
    "Select maps by rating",
    ["Any rating", "easy", "medium", "difficult"],
    index=0,
)

reviewer_filter = st.sidebar.selectbox(
    "Rating by",
    ["Any reviewer"] + REVIEWERS,
    index=0,
)

minimum_number_of_ratings = st.sidebar.slider(
    "Minimum number of ratings",
    min_value=0,
    max_value=len(REVIEWERS),
    value=0,
)

exclude_irrelevant = st.sidebar.checkbox(
    "Exclude maps marked irrelevant by anyone",
    value=True,
)

only_disagreement = st.sidebar.checkbox(
    "Only maps with disagreement among reviewers",
    value=False,
)

show_image_debug = st.sidebar.checkbox(
    "Show image debug info",
    value=False,
)

filtered_files = filter_maps_by_ratings(
    files=files,
    ratings_df=ratings_df,
    rating_filter=rating_filter,
    reviewer_filter=reviewer_filter,
    minimum_number_of_ratings=minimum_number_of_ratings,
    exclude_irrelevant=exclude_irrelevant,
    only_disagreement=only_disagreement,
)

st.sidebar.write(f"Matching maps: **{len(filtered_files)}**")

if not filtered_files:
    st.warning("No maps match the current filters.")
    st.stop()

selected_map_name = st.selectbox(
    "Choose map",
    [file["name"] for file in filtered_files],
)

selected_file = next(file for file in filtered_files if file["name"] == selected_map_name)

reset_annotation_state_for_new_map(selected_file["id"])

try:
    image_bytes = download_drive_image(selected_file["id"])
    original_image, display_image, scale = prepare_display_image(image_bytes)
except Exception as e:
    st.error("Could not download or prepare the selected image.")
    st.exception(e)
    st.stop()

original_width, original_height = original_image.size

st.write(f"Selected map: **{selected_file['name']}**")
st.caption(
    f"Original size: {original_width} × {original_height} px. "
    f"Displayed size: {display_image.size[0]} × {display_image.size[1]} px. "
    f"Display scale: {scale:.4f}."
)

rating_row = ratings_df[ratings_df["Map"] == selected_file["name"]]
if not rating_row.empty:
    with st.expander("Ratings for this map"):
        st.dataframe(rating_row, use_container_width=True)


tab_annotate, tab_gemini, tab_compare, tab_data = st.tabs(
    [
        "1. Human annotation",
        "2. Gemini detection",
        "3. Compare",
        "4. Data",
    ]
)


# ---------------------------------------------------------------------
# Tab 1: Human annotation
# ---------------------------------------------------------------------

with tab_annotate:
    st.subheader("Mark human text-area annotations")

    if not annotator:
        st.info("Select an annotator in the sidebar before saving annotations.")

    st.write(
        "Click once to set the first corner of a text area. "
        "Click a second time to set the opposite corner. "
        "Repeat this for each text-bearing region."
    )

    if show_image_debug:
        with st.expander("Image debugging", expanded=True):
            st.write("File:", selected_file["name"])
            st.write("File ID:", selected_file["id"])
            st.write("MIME type:", selected_file["mimeType"])
            st.write("Downloaded image bytes:", len(image_bytes))
            st.write("Original image mode:", original_image.mode)
            st.write("Original image size:", original_image.size)
            st.write("Display image mode:", display_image.mode)
            st.write("Display image size:", display_image.size)
            st.write("Scale:", scale)
            st.image(
                display_image,
                caption="Image preview used for annotation",
                use_container_width=True,
            )

    if st.session_state.current_box_start is None:
        st.caption("No active start point. Click the first corner of a text area.")
    else:
        st.caption(
            f"Start point selected at {st.session_state.current_box_start}. "
            "Click the opposite corner to complete the box."
        )
    
    annotation_preview = make_annotation_preview(
        display_image=display_image,
        boxes_display=st.session_state.human_boxes_display,
        current_box_start=st.session_state.current_box_start,
    )

    click = streamlit_image_coordinates(
        annotation_preview,
        key=f"annotation_image_{selected_file['id']}",
    )

    if click is not None:
        x_click = int(click["x"])
        y_click = int(click["y"])
    
        click_signature = (
            selected_file["id"],
            x_click,
            y_click,
        )
    
        if click_signature != st.session_state.last_processed_click:
            st.session_state.last_processed_click = click_signature
    
            if st.session_state.current_box_start is None:
                st.session_state.current_box_start = (x_click, y_click)
                st.rerun()
            else:
                x_start, y_start = st.session_state.current_box_start
    
                x_min = min(x_start, x_click)
                y_min = min(y_start, y_click)
                x_max = max(x_start, x_click)
                y_max = max(y_start, y_click)
    
                if x_max > x_min and y_max > y_min:
                    st.session_state.human_boxes_display.append(
                        {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        }
                    )
    
                st.session_state.current_box_start = None
                st.rerun()

    col_reset, col_undo, col_cancel = st.columns(3)

    with col_reset:
        if st.button("Clear boxes"):
            st.session_state.human_boxes_display = []
            st.session_state.current_box_start = None
            st.session_state.last_processed_click = None
            st.rerun()

    with col_undo:
        if st.button("Undo last box"):
            if st.session_state.human_boxes_display:
                st.session_state.human_boxes_display.pop()
            st.session_state.current_box_start = None
            st.session_state.last_processed_click = None
            st.rerun()

    with col_cancel:
        if st.button("Cancel current start point"):
            st.session_state.current_box_start = None
            st.session_state.last_processed_click = None
            st.rerun()

    if st.session_state.current_box_start is not None:
        st.info("First corner selected. Click the opposite corner to complete the box.")

    human_boxes = display_boxes_to_original_boxes(
        boxes_display=st.session_state.human_boxes_display,
        scale=scale,
    )

    st.write(f"Current boxes: **{len(human_boxes)}**")
    st.json(human_boxes)

    if st.button("Save human annotations", type="primary"):
        if not annotator:
            st.error("Select an annotator before saving.")
        elif not human_boxes:
            st.warning("No boxes to save.")
        else:
            save_human_annotation(
                sheet_url=sheet_url,
                map_name=selected_file["name"],
                file_id=selected_file["id"],
                annotator=annotator,
                image_width=original_width,
                image_height=original_height,
                boxes=human_boxes,
            )
            st.success(
                f"Saved {len(human_boxes)} human annotations for "
                f"{selected_file['name']}."
            )


# ---------------------------------------------------------------------
# Tab 2: Gemini detection
# ---------------------------------------------------------------------

with tab_gemini:
    st.subheader("Run Gemini text-area detection")

    model_name = st.text_input(
        "Gemini model",
        value=default_gemini_model,
        help=(
            "Keep this configurable. Use the Gemini model name available to "
            "your API account."
        ),
    )

    prompt_version = st.text_input(
        "Prompt version",
        value="v001",
    )

    with st.expander("Prompt preview"):
        st.code(
            build_gemini_prompt(
                image_width=original_width,
                image_height=original_height,
                prompt_version=prompt_version,
            ),
            language="text",
        )

    if st.button("Run Gemini on selected map", type="primary"):
        if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
            st.error("Missing Gemini API key in Streamlit secrets.")
        else:
            with st.spinner("Running Gemini..."):
                regions, raw_response = run_gemini_text_area_detection(
                    image_bytes=image_bytes,
                    mime_type=selected_file["mimeType"],
                    image_width=original_width,
                    image_height=original_height,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )

            save_gemini_prediction(
                sheet_url=sheet_url,
                map_name=selected_file["name"],
                file_id=selected_file["id"],
                model_name=model_name,
                prompt_version=prompt_version,
                image_width=original_width,
                image_height=original_height,
                regions=regions,
                raw_response=raw_response,
            )

            st.success(f"Saved {len(regions)} Gemini regions.")
            st.subheader("Gemini regions")
            st.json(regions)

            with st.expander("Raw Gemini response"):
                st.code(raw_response, language="json")


# ---------------------------------------------------------------------
# Tab 3: Compare
# ---------------------------------------------------------------------

with tab_compare:
    st.subheader("Compare human and Gemini detections")

    compare_annotator = st.selectbox(
        "Human annotation to compare",
        ["Latest for selected annotator", "Latest for any annotator"],
        index=0,
    )

    compare_model_name = st.text_input(
        "Gemini model to compare",
        value=default_gemini_model,
        key="compare_model_name",
    )

    compare_prompt_version = st.text_input(
        "Prompt version to compare",
        value="v001",
        key="compare_prompt_version",
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    selected_compare_annotator = (
        annotator if compare_annotator == "Latest for selected annotator" else None
    )

    latest_human = get_latest_human_annotation(
        sheet_url=sheet_url,
        map_name=selected_file["name"],
        annotator=selected_compare_annotator,
    )

    latest_gemini = get_latest_gemini_prediction(
        sheet_url=sheet_url,
        map_name=selected_file["name"],
        model_name=compare_model_name,
        prompt_version=compare_prompt_version,
    )

    if latest_human is None:
        st.warning("No saved human annotation found for this selection.")

    if latest_gemini is None:
        st.warning("No saved Gemini prediction found for this selection.")

    if latest_human is not None and latest_gemini is not None:
        saved_human_boxes = latest_human["boxes"]
        saved_gemini_regions = latest_gemini["regions"]

        result = evaluate_detections(
            human_boxes=saved_human_boxes,
            predicted_boxes=saved_gemini_regions,
            iou_threshold=iou_threshold,
        )

        st.subheader("Metrics")
        st.dataframe(make_metrics_df(result), use_container_width=True)

        st.subheader("Overlay")
        overlay = draw_boxes_on_image(
            image=display_image,
            human_boxes=saved_human_boxes,
            gemini_boxes=saved_gemini_regions,
            scale=scale,
        )

        st.image(
            overlay,
            caption="Red = human annotations; Blue = Gemini predictions",
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Human boxes")
            st.json(saved_human_boxes)

        with col_b:
            st.subheader("Gemini boxes")
            st.json(saved_gemini_regions)

        st.subheader("Matches")
        st.json(result["matches"])

        if st.button("Save evaluation result"):
            save_evaluation_result(
                sheet_url=sheet_url,
                map_name=selected_file["name"],
                file_id=selected_file["id"],
                annotator=str(latest_human.get("annotator", "")),
                model_name=compare_model_name,
                prompt_version=compare_prompt_version,
                iou_threshold=iou_threshold,
                result=result,
            )
            st.success("Saved evaluation result.")


# ---------------------------------------------------------------------
# Tab 4: Data
# ---------------------------------------------------------------------

with tab_data:
    st.subheader("Stored data")

    data_choice = st.radio(
        "Choose data table",
        [
            "Ratings",
            "Human annotations",
            "Gemini predictions",
        ],
        horizontal=True,
    )

    if data_choice == "Ratings":
        st.dataframe(
            add_rating_summary_columns(ratings_df),
            use_container_width=True,
        )

    elif data_choice == "Human annotations":
        human_df = load_human_annotations_df(sheet_url)
        st.dataframe(human_df, use_container_width=True)

    elif data_choice == "Gemini predictions":
        gemini_df = load_gemini_predictions_df(sheet_url)
        st.dataframe(gemini_df, use_container_width=True)
