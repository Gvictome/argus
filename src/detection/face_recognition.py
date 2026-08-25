"""
Face recognition service for ARGUS.

Uses ArcFace (via the `insightface` library) to generate 512-d face
embeddings and match them against a database of known faces.

Pipeline:
    1. Detect face locations using InsightFace's RetinaFace detector.
    2. Compute 512-d ArcFace embedding for each detected face.
    3. Compare embeddings (cosine similarity) against known faces in DB.
    4. Return identity (face_id + name) or "unknown" for each face.

Dependencies:
    pip install insightface onnxruntime numpy opencv-python
"""

import logging
import pickle
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _FaceAnalysis = None
    _INSIGHTFACE_AVAILABLE = False
    logger.warning(
        "insightface not installed — FaceRecognitionService unavailable. "
        "Install with: pip install insightface onnxruntime"
    )


@dataclass
class FaceMatch:
    """Result of a face recognition attempt."""
    face_id: Optional[str]
    name: Optional[str]
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h (top-left origin)
    embedding: np.ndarray


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FaceRecognitionService:
    """
    ArcFace-based face enrollment, embedding generation, and identity matching.

    Known faces are stored in the Database (faces table) with their 512-d
    ArcFace embeddings serialized as pickle blobs.  An in-memory cache is
    maintained for fast matching during live detection.
    """

    def __init__(self, db, similarity_threshold: float = 0.4, det_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            db: Database instance (src.database.Database) — must be initialized.
            similarity_threshold: Min cosine similarity to consider a match.
                                  ArcFace typical range: 0.3–0.5 depending on use case.
            det_size: Input size for the face detector (width, height).
        """
        if not _INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface is required for ArcFace face recognition. "
                "Install with: pip install insightface onnxruntime"
            )
        self.db = db
        self.similarity_threshold = similarity_threshold

        # Initialize InsightFace app (downloads models on first run)
        self._app = _FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)
        logger.info("ArcFace (buffalo_l) initialized with det_size=%s", det_size)

        # In-memory cache: face_id -> (name, embedding)
        self._known_faces: Dict[str, Tuple[str, np.ndarray]] = {}
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        """Load all known face embeddings from the database into memory."""
        rows = self.db.list_faces()
        self._known_faces.clear()

        for row in rows:
            face_id = row["id"]
            name = row["name"]
            full = self.db.get_face(face_id)
            if full and full.get("embedding"):
                try:
                    embedding = pickle.loads(full["embedding"])
                    self._known_faces[face_id] = (name, embedding)
                except Exception as e:
                    logger.warning("Failed to load embedding for %s: %s", face_id, e)

        logger.info("Loaded %d known faces into cache", len(self._known_faces))

    def _detect_faces(self, frame: np.ndarray):
        """Run InsightFace detection + embedding on a BGR frame."""
        return self._app.get(frame)

    def enroll_face(
        self,
        frame: np.ndarray,
        name: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """
        Enroll a new face from a frame.

        If bbox is provided, crops to that region first; otherwise detects
        the largest face in the full frame.

        Args:
            frame: BGR numpy array.
            name: Name/label for this person.
            bbox: Optional (x, y, w, h) crop region.

        Returns:
            face_id if enrollment succeeds, None if no face found.
        """
        if bbox is not None:
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            faces = self._detect_faces(crop)
        else:
            faces = self._detect_faces(frame)

        if not faces:
            logger.warning("No face found in frame for enrollment")
            return None

        # Pick the largest face by bounding box area
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding  # 512-d normalized ArcFace vector

        face_id = str(uuid.uuid4())
        embedding_blob = pickle.dumps(embedding)

        self.db.add_face(face_id, name, embedding_blob)
        self._known_faces[face_id] = (name, embedding)

        logger.info("Enrolled face '%s' as %s", name, face_id)
        return face_id

    def remove_face(self, face_id: str) -> bool:
        """Remove a face from the database and cache."""
        self.db.delete_face(face_id)
        removed = self._known_faces.pop(face_id, None) is not None
        if removed:
            logger.info("Removed face %s", face_id)
        return removed

    def reset(self) -> int:
        """
        Remove every enrolled face from the database and the cache.

        Showcase P1-6: over a multi-hour demo the known-faces database grows
        with every passer-by who gets enrolled. An unpruned database degrades
        match quality and eventually resolves the wrong identity in front of
        a judge, which is a worse failure than not recognizing anyone.

        Returns:
            Number of faces removed.
        """
        removed = self.db.clear_faces()
        self._known_faces.clear()
        logger.info("Reset known-faces database — removed %d faces", removed)
        return removed

    def recognize(self, frame: np.ndarray) -> List[FaceMatch]:
        """
        Detect and identify all faces in a frame.

        Args:
            frame: BGR numpy array.

        Returns:
            List of FaceMatch results — one per detected face.
            Unknown faces have face_id=None and name=None.
        """
        faces = self._detect_faces(frame)
        if not faces:
            return []

        known_ids = list(self._known_faces.keys())
        known_embeddings = [self._known_faces[fid][1] for fid in known_ids]

        results: List[FaceMatch] = []

        for face in faces:
            # InsightFace bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            embedding = face.normed_embedding

            face_id = None
            name = None
            confidence = 0.0

            if known_embeddings:
                similarities = [
                    _cosine_similarity(embedding, known_emb)
                    for known_emb in known_embeddings
                ]
                best_idx = int(np.argmax(similarities))
                best_sim = similarities[best_idx]

                if best_sim >= self.similarity_threshold:
                    face_id = known_ids[best_idx]
                    name = self._known_faces[face_id][0]
                    confidence = best_sim
                    self.db.update_face_seen(face_id)

            results.append(FaceMatch(
                face_id=face_id,
                name=name,
                confidence=confidence,
                bbox=(x, y, w, h),
                embedding=embedding,
            ))

        return results

    def recognize_in_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> List[FaceMatch]:
        """
        Run face recognition only within a bounding box region.

        Useful when YOLO has already detected a person and we want to
        identify just that crop.

        Args:
            frame: Full BGR frame.
            bbox: (x, y, w, h) region of interest.

        Returns:
            List of FaceMatch results found within the region.
        """
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        matches = self.recognize(crop)

        # Adjust bbox coordinates back to full-frame space
        for m in matches:
            mx, my, mw, mh = m.bbox
            m.bbox = (mx + x1, my + y1, mw, mh)

        return matches

    def list_known_faces(self) -> List[Dict]:
        """Return summary of all enrolled faces."""
        return [
            {"face_id": fid, "name": name}
            for fid, (name, _) in self._known_faces.items()
        ]

    def reload(self) -> None:
        """Reload known faces from the database (e.g. after external changes)."""
        self._load_known_faces()
