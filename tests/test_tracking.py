# tests/test_tracking.py


import pytest

from omnimcp.tracking import SimpleElementTracker

# Assuming types and tracker are in these locations
from omnimcp.types import Bounds, UIElement

# --- Test Helpers ---


def make_element(
    id: int,
    type: str = "button",
    content: str = "Test",
    bounds: Bounds = (0.1, 0.1, 0.05, 0.05),  # Default small box
    **kwargs,  # Allow adding other attributes if needed later
) -> UIElement:
    """Helper to create UIElement instances for tests."""
    return UIElement(
        id=id,
        type=type,
        content=content,
        bounds=bounds,
        confidence=0.95,  # Default confidence
        attributes=kwargs.get("attributes", {}),
    )


# --- Test Fixtures ---


@pytest.fixture
def tracker() -> SimpleElementTracker:
    """Provides a fresh SimpleElementTracker instance for each test."""
    # Use default thresholds for most tests, can override later if needed
    return SimpleElementTracker(miss_threshold=3, matching_threshold=0.1)


# --- Test Cases ---


def test_tracker_initialization(tracker: SimpleElementTracker):
    """Test that the tracker initializes correctly."""
    assert tracker.tracked_elements == {}
    assert tracker.next_track_id_counter == 0
    assert tracker.miss_threshold == 3
    assert tracker.match_threshold_sq == pytest.approx(0.1**2)


def test_update_empty_tracker_with_elements(tracker: SimpleElementTracker):
    """Test adding elements to an empty tracker."""
    frame_num = 1
    elements = [
        make_element(id=0, type="button", content="OK", bounds=(0.1, 0.1, 0.1, 0.1)),
        make_element(id=1, type="text", content="Label", bounds=(0.3, 0.3, 0.2, 0.05)),
    ]

    tracked_list = tracker.update(elements, frame_num)

    assert len(tracked_list) == 2
    assert len(tracker.tracked_elements) == 2
    assert tracker.next_track_id_counter == 2

    # Check track details (assuming sequential track_id assignment)
    track0 = tracker.tracked_elements.get("track_0")
    track1 = tracker.tracked_elements.get("track_1")

    assert track0 is not None
    assert track1 is not None

    # Verify based on which element got which track_id (implementation dependent,
    # but assume stable order for now)
    # Let's assume track_0 corresponds to element 0, track_1 to element 1
    if track0.latest_element.id == 0:
        ok_track, label_track = track0, track1
    else:
        ok_track, label_track = track1, track0

    assert ok_track.track_id in ["track_0", "track_1"]
    assert ok_track.latest_element is not None
    assert ok_track.latest_element.id == 0  # Check if correct element is linked
    assert ok_track.latest_element.content == "OK"
    assert ok_track.consecutive_misses == 0
    assert ok_track.last_seen_frame == frame_num

    assert label_track.track_id in ["track_0", "track_1"]
    assert label_track.latest_element is not None
    assert label_track.latest_element.id == 1
    assert label_track.latest_element.content == "Label"
    assert label_track.consecutive_misses == 0
    assert label_track.last_seen_frame == frame_num

    # Ensure the returned list matches internal state (order might differ)
    assert len(tracked_list) == len(tracker.tracked_elements)
    assert {t.track_id for t in tracked_list} == set(tracker.tracked_elements.keys())


def test_update_empty_current_elements(tracker: SimpleElementTracker):
    """Test updating with no elements when tracks exist."""
    # Frame 1: Add initial elements
    frame1_elements = [make_element(id=0, bounds=(0.1, 0.1, 0.1, 0.1))]
    tracker.update(frame1_elements, 1)
    assert len(tracker.tracked_elements) == 1
    initial_track = tracker.tracked_elements["track_0"]
    assert initial_track.consecutive_misses == 0
    assert initial_track.latest_element is not None

    # Frame 2: Update with empty list
    frame2_elements: list[UIElement] = []
    tracked_list = tracker.update(frame2_elements, 2)

    assert len(tracked_list) == 1  # Track still exists
    assert len(tracker.tracked_elements) == 1
    updated_track = tracker.tracked_elements["track_0"]

    assert updated_track.track_id == "track_0"
    assert updated_track.latest_element is None  # Marked as missing
    assert updated_track.consecutive_misses == 1  # Miss count incremented
    assert updated_track.last_seen_frame == 1  # Last seen frame remains the same


def test_update_perfect_persistence(tracker: SimpleElementTracker):
    """Test elements staying in the same place."""
    frame1 = 1
    elements1 = [
        make_element(id=10, type="button", content="OK", bounds=(0.1, 0.1, 0.1, 0.1))
    ]
    tracker.update(elements1, frame1)
    assert "track_0" in tracker.tracked_elements
    assert tracker.tracked_elements["track_0"].last_seen_frame == frame1
    assert tracker.tracked_elements["track_0"].consecutive_misses == 0

    frame2 = 2
    # Use different element ID but same properties and position
    elements2 = [
        make_element(id=20, type="button", content="OK", bounds=(0.1, 0.1, 0.1, 0.1))
    ]
    tracked_list = tracker.update(elements2, frame2)

    assert len(tracked_list) == 1
    assert len(tracker.tracked_elements) == 1
    # Should still be track_0, assuming matching works
    assert "track_0" in tracker.tracked_elements
    persisted_track = tracker.tracked_elements["track_0"]

    assert persisted_track.latest_element is not None
    assert (
        persisted_track.latest_element.id == 20
    )  # ID updated to current frame's element
    assert persisted_track.consecutive_misses == 0  # No misses
    assert persisted_track.last_seen_frame == frame2  # Last seen updated


def test_update_disappearance_and_pruning(tracker: SimpleElementTracker):
    """Test element disappearing and getting pruned after threshold."""
    tracker = SimpleElementTracker(miss_threshold=2)  # Lower threshold for test

    # Frame 1: Element appears
    tracker.update([make_element(id=0, bounds=(0.5, 0.5, 0.1, 0.1))], 1)
    assert "track_0" in tracker.tracked_elements

    # Frame 2: Element disappears
    tracker.update([], 2)
    assert tracker.tracked_elements["track_0"].consecutive_misses == 1
    assert tracker.tracked_elements["track_0"].latest_element is None

    # Frame 3: Element still disappeared (reaches miss_threshold)
    tracker.update([], 3)
    # Track should be pruned *after* this update completes
    assert "track_0" not in tracker.tracked_elements
    assert len(tracker.tracked_elements) == 0


def test_update_appearance(tracker: SimpleElementTracker):
    """Test a new element appearing alongside a persistent one."""
    # Frame 1
    elements1 = [
        make_element(id=0, type="button", content="A", bounds=(0.1, 0.1, 0.1, 0.1))
    ]
    tracker.update(elements1, 1)
    assert "track_0" in tracker.tracked_elements
    assert len(tracker.tracked_elements) == 1

    # Frame 2: Element A persists, Element B appears
    elements2 = [
        make_element(
            id=10, type="button", content="A", bounds=(0.1, 0.1, 0.1, 0.1)
        ),  # Persistent
        make_element(
            id=11, type="button", content="B", bounds=(0.3, 0.3, 0.1, 0.1)
        ),  # New
    ]
    tracked_list = tracker.update(elements2, 2)

    assert len(tracked_list) == 2
    assert len(tracker.tracked_elements) == 2
    assert "track_0" in tracker.tracked_elements  # Original track persists
    assert "track_1" in tracker.tracked_elements  # New track created

    track_a = tracker.tracked_elements["track_0"]
    track_b = tracker.tracked_elements["track_1"]

    assert track_a.latest_element is not None
    assert track_a.latest_element.id == 10  # Updated element ID
    assert track_a.consecutive_misses == 0
    assert track_a.last_seen_frame == 2

    assert track_b.latest_element is not None
    assert track_b.latest_element.id == 11
    assert track_b.latest_element.content == "B"
    assert track_b.consecutive_misses == 0
    assert track_b.last_seen_frame == 2


# --- TODO: Add More Complex Tests ---
# - test_positional_shift_within_threshold
# - test_positional_shift_outside_threshold
# - test_type_mismatch_same_position
# - test_multiple_matches_scenario (ensure optimal assignment works)
# - test_with_invalid_bounds (ensure helper handles None center)
