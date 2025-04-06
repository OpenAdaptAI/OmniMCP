# omnimcp/tracking.py
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from loguru import logger

# Assuming UIElement and ElementTrack are defined in omnimcp.types
try:
    from omnimcp.types import UIElement, ElementTrack, Bounds
except ImportError:
    # This allows standalone testing/linting but relies on installation for runtime
    logger.warning(
        "Could not import types directly from omnimcp.types. Relying on installed package."
    )
    UIElement = dict  # type: ignore
    ElementTrack = dict  # type: ignore
    Bounds = tuple  # type: ignore


def _get_bounds_center(bounds: Bounds) -> Optional[Tuple[float, float]]:
    """Calculate the center (relative coords 0.0-1.0) of a bounding box."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        logger.warning(f"Invalid bounds format: {bounds}. Cannot calculate center.")
        return None
    x, y, w, h = bounds
    if w < 0 or h < 0:
        logger.warning(
            f"Invalid bounds dimensions (w={w}, h={h}). Cannot calculate center."
        )
        return None
    return x + w / 2, y + h / 2


class SimpleElementTracker:
    """
    Tracks UI elements across frames using optimal assignment based on type and proximity.
    Assigns persistent track_ids.
    """

    def __init__(self, miss_threshold: int = 3, matching_threshold: float = 0.1):
        """
        Args:
            miss_threshold: Number of consecutive misses before pruning a track.
            matching_threshold: Relative distance threshold for matching centers.
        """
        self.tracked_elements: Dict[str, ElementTrack] = {}  # track_id -> ElementTrack
        self.next_track_id_counter: int = 0
        self.miss_threshold = miss_threshold
        self.match_threshold_sq = (
            matching_threshold**2
        )  # Use squared distance for efficiency
        logger.info(
            f"SimpleElementTracker initialized (miss_thresh={miss_threshold}, match_dist_sq={self.match_threshold_sq:.4f})."
        )

    def _generate_track_id(self) -> str:
        """Generates a unique, sequential track ID."""
        track_id = f"track_{self.next_track_id_counter}"
        self.next_track_id_counter += 1
        return track_id

    def _match_elements(self, current_elements: List[UIElement]) -> Dict[int, str]:
        """
        Performs optimal assignment matching between current elements and active tracks
        using the Hungarian algorithm (linear_sum_assignment).

        Args:
            current_elements: List of UIElements detected in the current frame.

        Returns:
            Dict[int, str]: Mapping from current_element.id to matched track_id.
        """
        # Filter out invalid elements and prepare data for matching
        valid_current_elements = [
            el for el in current_elements if _get_bounds_center(el.bounds) is not None
        ]
        active_tracks = [
            track
            for track in self.tracked_elements.values()
            if track.latest_element is not None
            and _get_bounds_center(track.latest_element.bounds) is not None
        ]

        if not valid_current_elements or not active_tracks:
            logger.debug("No valid current elements or active tracks to match.")
            return {}

        current_ids = [el.id for el in valid_current_elements]
        current_types = [el.type for el in valid_current_elements]
        current_centers = np.array(
            [_get_bounds_center(el.bounds) for el in valid_current_elements]
        )

        track_ids = [track.track_id for track in active_tracks]
        track_types = [
            track.latest_element.type for track in active_tracks
        ]  # Assumes latest_element is not None
        track_centers = np.array(
            [_get_bounds_center(track.latest_element.bounds) for track in active_tracks]
        )  # Assumes latest_element is not None

        # Calculate Cost Matrix (Squared Euclidean Distance)
        # Rows: current elements, Cols: active tracks
        cost_matrix = cdist(current_centers, track_centers, metric="sqeuclidean")

        # Apply Constraints (Type Mismatch & Distance Threshold)
        infinity_cost = 1e8  # Use a large number
        num_current, num_tracks = cost_matrix.shape

        for i in range(num_current):
            for j in range(num_tracks):
                # High cost if types don't match
                if current_types[i] != track_types[j]:
                    cost_matrix[i, j] = infinity_cost
                # High cost if distance exceeds threshold
                elif cost_matrix[i, j] > self.match_threshold_sq:
                    cost_matrix[i, j] = infinity_cost

        # Optimal Assignment using Hungarian Algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.error(
                f"Error during linear_sum_assignment: {e}. Cost matrix shape: {cost_matrix.shape}"
            )
            return {}

        # Create Mapping from Valid Assignments
        assignment_mapping: Dict[int, str] = {}  # current_element_id -> track_id
        valid_matches_count = 0
        for r, c in zip(row_ind, col_ind):
            # Check if the assignment cost is valid (below infinity_cost)
            if cost_matrix[r, c] < infinity_cost:
                current_element_id = current_ids[r]
                track_id = track_ids[c]
                assignment_mapping[current_element_id] = track_id
                valid_matches_count += 1

        logger.debug(
            f"Matching: Found {valid_matches_count} valid assignments using linear_sum_assignment."
        )
        return assignment_mapping

    def update(
        self, current_elements: List[UIElement], frame_number: int
    ) -> List[ElementTrack]:
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

        tracks_to_prune: List[str] = []
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
                    # Defensive coding for edge cases where mapping might be inconsistent
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
                f"Pruning track {track_id} after {self.tracked_elements.get(track_id, ElementTrack(track_id=track_id)).consecutive_misses} misses."
            )  # Safely access misses
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
