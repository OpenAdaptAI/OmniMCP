# omnimcp/tracking.py

from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from loguru import logger

from omnimcp.types import UIElement, ElementTrack, Bounds


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
    Tracks UI elements across frames using optimal assignment based on type,
    center proximity, and size similarity. Assigns persistent track_ids.
    """

    def __init__(
        self,
        miss_threshold: int = 3,
        matching_threshold: float = 0.1,
        size_rel_threshold: float = 0.3,
    ):
        """
        Args:
            miss_threshold: Number of consecutive misses before pruning a track.
            matching_threshold: Relative distance threshold for matching centers.
            size_rel_threshold: Relative size difference threshold for width/height.
        """
        self.tracked_elements: Dict[str, ElementTrack] = {}  # track_id -> ElementTrack
        self.next_track_id_counter: int = 0
        self.miss_threshold = miss_threshold
        self.match_threshold_sq = matching_threshold**2  # Use squared distance
        self.size_rel_threshold = size_rel_threshold
        logger.info(
            f"SimpleElementTracker initialized (miss_thresh={miss_threshold}, "
            f"match_dist_sq={self.match_threshold_sq:.4f}, "
            f"size_rel_thresh={self.size_rel_threshold:.2f})."
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
        # Filter out elements with invalid bounds and prepare data
        valid_current_elements = []
        current_centers_list = []
        for el in current_elements:
            center = _get_bounds_center(el.bounds)
            if center is not None:
                valid_current_elements.append(el)
                current_centers_list.append(center)

        active_tracks = []
        track_centers_list = []
        for track in self.tracked_elements.values():
            if track.latest_element:  # Check if track has a valid last known element
                center = _get_bounds_center(track.latest_element.bounds)
                if center is not None:
                    active_tracks.append(track)
                    track_centers_list.append(center)

        if not valid_current_elements or not active_tracks:
            logger.debug("No valid current elements or active tracks to match.")
            return {}

        # Extract properties for cost calculation
        current_ids = [el.id for el in valid_current_elements]
        current_types = [el.type for el in valid_current_elements]
        current_bounds_list = [el.bounds for el in valid_current_elements]
        current_centers = np.array(current_centers_list)

        track_ids = [track.track_id for track in active_tracks]
        track_types = [
            track.latest_element.type for track in active_tracks
        ]  # Safe due to filtering above
        track_bounds_list = [
            track.latest_element.bounds for track in active_tracks
        ]  # Safe due to filtering above
        track_centers = np.array(track_centers_list)

        # Calculate Cost Matrix (Squared Euclidean Distance)
        cost_matrix = cdist(current_centers, track_centers, metric="sqeuclidean")

        # Apply Constraints (Type Mismatch, Distance Threshold, Size Threshold)
        infinity_cost = 1e8  # Use a large number for invalid assignments
        num_current, num_tracks = cost_matrix.shape
        epsilon = 1e-6  # Avoid division by zero

        for i in range(num_current):
            for j in range(num_tracks):
                # --- Type Constraint ---
                if current_types[i] != track_types[j]:
                    cost_matrix[i, j] = infinity_cost
                    continue

                # --- Distance Constraint ---
                # Check if distance cost already exceeds threshold (slightly redundant but explicit)
                if cost_matrix[i, j] > self.match_threshold_sq:
                    cost_matrix[i, j] = infinity_cost
                    continue

                # --- Size Constraint ---
                curr_w, curr_h = current_bounds_list[i][2], current_bounds_list[i][3]
                track_w, track_h = track_bounds_list[j][2], track_bounds_list[j][3]

                # Use max dimensions for relative comparison denominator
                max_w = max(curr_w, track_w, epsilon)
                max_h = max(curr_h, track_h, epsilon)

                rel_width_diff = abs(curr_w - track_w) / max_w
                rel_height_diff = abs(curr_h - track_h) / max_h

                if (
                    rel_width_diff > self.size_rel_threshold
                    or rel_height_diff > self.size_rel_threshold
                ):
                    cost_matrix[i, j] = infinity_cost
                    continue  # Element size differs too much

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
            # Check if the assignment cost is valid
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
        Includes logic for handling misses and pruning tracks.

        Args:
            current_elements: List of UIElements detected in the current frame.
            frame_number: The current step/frame number.

        Returns:
            A list of all currently active ElementTrack objects.
        """
        current_element_map = {el.id: el for el in current_elements}
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
                    # Defensive coding for edge cases
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
        pruned_count = 0
        for track_id in tracks_to_prune:
            if track_id in self.tracked_elements:
                misses = self.tracked_elements[track_id].consecutive_misses
                del self.tracked_elements[track_id]
                logger.debug(f"Pruning track {track_id} after {misses} misses.")
                pruned_count += 1
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} tracks.")

        # Add tracks for new, unmatched elements
        new_count = 0
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
                new_count += 1
        if new_count > 0:
            logger.debug(f"Created {new_count} new tracks.")

        # Return the current list of all tracked elements' state
        return list(self.tracked_elements.values())
