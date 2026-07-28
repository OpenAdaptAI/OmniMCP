# omnimcp/tracking.py

# Use typing_extensions for Self if needed for older Python versions
# from typing_extensions import Self
# Added Scipy for matching
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback or warning needed if scipy is critical
    import warnings

    warnings.warn(
        "Scipy not found. Tracking matching will be disabled or use a fallback."
    )


# Assuming UIElement and ElementTrack are defined in omnimcp.types
try:
    from omnimcp.types import Bounds, ElementTrack, UIElement
except ImportError:
    print("Warning: Could not import types from omnimcp.types")
    UIElement = dict  # type: ignore
    ElementTrack = dict  # type: ignore
    Bounds = tuple  # type: ignore

# Assuming logger is setup elsewhere and accessible, or use standard logging
# from omnimcp.utils import logger
import logging

logger = logging.getLogger(__name__)


# Helper Function (can stay here or move to utils)
def _get_bounds_center(bounds: Bounds) -> tuple[float, float] | None:
    """Calculate the center (relative coords) of a bounding box."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        logger.warning(
            f"Invalid bounds format received: {bounds}. Cannot calculate center."
        )
        return None
    x, y, w, h = bounds
    # Ensure w and h are non-negative
    if w < 0 or h < 0:
        logger.warning(
            f"Invalid bounds dimensions (w={w}, h={h}). Cannot calculate center."
        )
        return None
    return x + w / 2, y + h / 2


class SimpleElementTracker:
    """
    Basic element tracking across frames based on type and proximity using optimal assignment.
    Assigns persistent track_ids.
    """

    def __init__(
        self, miss_threshold: int = 3, matching_threshold: float = 0.1
    ):  # Increased threshold slightly
        """
        Args:
            miss_threshold: How many consecutive misses before pruning a track.
            matching_threshold: Relative distance threshold for matching centers.
        """
        if not SCIPY_AVAILABLE:
            # Optionally raise an error or disable tracking features
            logger.error(
                "Scipy is required for SimpleElementTracker matching logic but not installed."
            )
            # raise ImportError("Scipy is required for SimpleElementTracker")
        self.tracked_elements: dict[str, ElementTrack] = {}  # track_id -> ElementTrack
        self.next_track_id_counter: int = 0
        self.miss_threshold = miss_threshold
        # Store squared threshold for efficiency
        self.match_threshold_sq = matching_threshold**2
        logger.info(
            f"SimpleElementTracker initialized (miss_thresh={miss_threshold}, match_dist_sq={self.match_threshold_sq:.4f})."
        )

    def _generate_track_id(self) -> str:
        """Generates a unique track ID."""
        track_id = f"track_{self.next_track_id_counter}"
        self.next_track_id_counter += 1
        return track_id

    def _match_elements(self, current_elements: list[UIElement]) -> dict[int, str]:
        """
        Performs optimal assignment matching between current elements and active tracks.

        Args:
            current_elements: List of UIElements detected in the current frame.

        Returns:
            Dict[int, str]: A mapping from current_element.id to matched track_id.
                           Only includes elements that were successfully matched.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available, skipping matching.")
            return {}
        if not current_elements or not self.tracked_elements:
            return {}  # Nothing to match

        # --- Prepare Data for Matching ---
        active_tracks = [
            track
            for track in self.tracked_elements.values()
            if track.latest_element is not None  # Only match tracks currently visible
        ]
        if not active_tracks:
            return {}  # No active tracks to match against

        # current_element_map = {el.id: el for el in current_elements}
        # track_map = {track.track_id: track for track in active_tracks}

        # Get centers and types for cost calculation
        current_centers = np.array(
            [
                _get_bounds_center(el.bounds)
                for el in current_elements
                if _get_bounds_center(el.bounds) is not None  # Filter invalid bounds
            ]
        )
        current_types = [
            el.type
            for el in current_elements
            if _get_bounds_center(el.bounds) is not None
        ]
        current_ids_valid = [
            el.id
            for el in current_elements
            if _get_bounds_center(el.bounds) is not None
        ]

        track_centers = np.array(
            [
                _get_bounds_center(track.latest_element.bounds)
                for track in active_tracks
                if track.latest_element
                and _get_bounds_center(track.latest_element.bounds) is not None
            ]
        )
        track_types = [
            track.latest_element.type
            for track in active_tracks
            if track.latest_element
            and _get_bounds_center(track.latest_element.bounds) is not None
        ]
        track_ids_valid = [
            track.track_id
            for track in active_tracks
            if track.latest_element
            and _get_bounds_center(track.latest_element.bounds) is not None
        ]

        if current_centers.size == 0 or track_centers.size == 0:
            logger.debug("No valid centers for matching.")
            return {}  # Cannot match if no valid centers

        # --- Calculate Cost Matrix (Squared Euclidean Distance) ---
        # Cost matrix: rows = current elements, cols = active tracks
        cost_matrix = cdist(current_centers, track_centers, metric="sqeuclidean")

        # --- Apply Constraints (Type Mismatch & Distance Threshold) ---
        infinity_cost = float("inf")
        num_current, num_tracks = cost_matrix.shape

        for i in range(num_current):
            for j in range(num_tracks):
                # Infinite cost if types don't match
                if (
                    current_types[i] != track_types[j]
                    or cost_matrix[i, j] > self.match_threshold_sq
                ):
                    cost_matrix[i, j] = infinity_cost

        # --- Optimal Assignment using Hungarian Algorithm ---
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.error(
                f"Error during linear_sum_assignment: {e}. Cost matrix shape: {cost_matrix.shape}"
            )
            return {}

        # --- Create Mapping from Valid Assignments ---
        assignment_mapping: dict[int, str] = {}  # current_element_id -> track_id
        valid_matches_count = 0
        for r, c in zip(row_ind, col_ind):
            # Check if the assignment cost is valid (not infinity)
            if cost_matrix[r, c] < infinity_cost:
                current_element_id = current_ids_valid[r]
                track_id = track_ids_valid[c]
                assignment_mapping[current_element_id] = track_id
                valid_matches_count += 1

        logger.debug(f"Matching: Found {valid_matches_count} valid assignments.")
        return assignment_mapping

    def update(
        self, current_elements: list[UIElement], frame_number: int
    ) -> list[ElementTrack]:
        """
        Updates tracks based on current detections using optimal assignment matching.

        Args:
            current_elements: List of UIElements detected in the current frame.
            frame_number: The current step/frame number.

        Returns:
            A list of all currently active ElementTrack objects (including missed ones).
        """
        current_element_map = {el.id: el for el in current_elements}

        # Get the mapping: current_element_id -> track_id
        assignment_mapping = self._match_elements(current_elements)

        matched_current_element_ids = set(assignment_mapping.keys())
        matched_track_ids = set(assignment_mapping.values())

        tracks_to_prune: list[str] = []
        # Update existing tracks based on matches
        for track_id, track in self.tracked_elements.items():
            if track_id in matched_track_ids:
                # Find the current element that matched this track
                matched_elem_id = next(
                    (
                        curr_id
                        for curr_id, t_id in assignment_mapping.items()
                        if t_id == track_id
                    ),
                    None,
                )

                if (
                    matched_elem_id is not None
                    and matched_elem_id in current_element_map
                ):
                    # Matched successfully
                    track.latest_element = current_element_map[matched_elem_id]
                    track.consecutive_misses = 0
                    track.last_seen_frame = frame_number
                else:
                    # Match found in assignment but element missing from map (should not happen ideally)
                    logger.warning(
                        f"Track {track_id} matched but element ID {matched_elem_id} not found in current_element_map. Treating as miss."
                    )
                    track.latest_element = None
                    track.consecutive_misses += 1
                    logger.debug(
                        f"Track {track_id} treated as missed frame {frame_number}. Consecutive misses: {track.consecutive_misses}"
                    )
                    if track.consecutive_misses >= self.miss_threshold:
                        tracks_to_prune.append(track_id)
            else:
                # Track was not matched in the current frame
                track.latest_element = None
                track.consecutive_misses += 1
                logger.debug(
                    f"Track {track_id} missed frame {frame_number}. Consecutive misses: {track.consecutive_misses}"
                )
                # Check for pruning AFTER incrementing misses
                if track.consecutive_misses >= self.miss_threshold:
                    tracks_to_prune.append(track_id)

        # Prune tracks marked for deletion
        for track_id in tracks_to_prune:
            logger.debug(
                f"Pruning track {track_id} after {self.tracked_elements[track_id].consecutive_misses} misses."
            )
            if track_id in self.tracked_elements:
                del self.tracked_elements[track_id]

        # Add tracks for new, unmatched elements
        for element_id, element in current_element_map.items():
            if element_id not in matched_current_element_ids:
                # Ensure element has valid bounds before creating track
                if _get_bounds_center(element.bounds) is None:
                    logger.debug(
                        f"Skipping creation of track for element ID {element_id} due to invalid bounds."
                    )
                    continue

                new_track_id = self._generate_track_id()
                new_track = ElementTrack(
                    track_id=new_track_id,
                    latest_element=element,
                    consecutive_misses=0,
                    last_seen_frame=frame_number,
                )
                self.tracked_elements[new_track_id] = new_track
                logger.debug(
                    f"Created new track {new_track_id} for element ID {element_id}"
                )

        # Return the current list of all tracked elements' state
        return list(self.tracked_elements.values())
