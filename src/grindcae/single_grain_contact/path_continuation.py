"""Small deterministic pseudo-arclength kernel for constrained event paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import NDArray


class PathContinuationError(RuntimeError):
    """Raised when a constrained path correction cannot reach equilibrium."""


@dataclass(frozen=True, slots=True)
class AcceptedPathReference:
    """Physical direction and provenance, independent of active-mesh numbering."""
    background_node_ids: tuple[int, ...]
    displacement_direction_m: tuple[tuple[float, float], ...]
    marker_direction: float
    displacement_scale_m: float
    marker_scale_m: float
    source_markers: tuple[float, float]
    source: str

    def __post_init__(self):
        values=np.asarray(self.displacement_direction_m,dtype=float)
        if (values.shape!=(len(self.background_node_ids),2) or not np.all(np.isfinite(values))
            or len(set(self.background_node_ids))!=len(self.background_node_ids)
            or any(type(node) is not int or node<0 for node in self.background_node_ids)):
            raise ValueError('invalid accepted path background direction')
        if (not all(math.isfinite(v) for v in (self.marker_direction,*self.source_markers))
            or not all(math.isfinite(v) and v>0 for v in (self.displacement_scale_m,self.marker_scale_m))
            or self.source not in ('accepted_increment','accepted_augmented_tangent')):
            raise ValueError('invalid accepted path scales or provenance')

    def mapped_vector(self,background_nodes,free_dofs,*,displacement_scale_m,marker_scale_m):
        if not all(math.isfinite(v) and v>0 for v in (displacement_scale_m,marker_scale_m)):
            raise ValueError('positive mapped path scales required')
        previous=dict(zip(self.background_node_ids,self.displacement_direction_m,strict=True))
        if any(int(node) not in previous for node in background_nodes):
            raise ValueError('path map introduces unknown background identity')
        physical=np.array([previous[int(node)] for node in background_nodes],dtype=float).reshape(-1)
        dofs=np.asarray(free_dofs,dtype=int)
        if dofs.ndim!=1 or np.any(dofs<0) or np.any(dofs>=physical.size):
            raise ValueError('invalid mapped free degrees of freedom')
        return np.concatenate((physical[dofs]/displacement_scale_m,
            [self.marker_direction*marker_scale_m/displacement_scale_m]))


ResidualFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
JacobianFunction = Callable[
    [NDArray[np.float64], float],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


@dataclass(frozen=True, slots=True)
class PseudoArcLengthSettings:
    arc_length: float
    maximum_corrections: int
    residual_tolerance: float
    constraint_tolerance: float
    correction_tolerance: float = math.inf

    def __post_init__(self) -> None:
        if not math.isfinite(self.arc_length) or self.arc_length <= 0.0:
            raise ValueError("arc_length must be finite and positive")
        if not isinstance(self.maximum_corrections, int) or isinstance(
            self.maximum_corrections, bool
        ) or self.maximum_corrections < 1:
            raise ValueError("maximum_corrections must be a positive integer")
        for name in (
            "residual_tolerance",
            "constraint_tolerance",
            "correction_tolerance",
        ):
            value = float(getattr(self, name))
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and not NaN")


@dataclass(frozen=True, slots=True)
class PathState:
    state: NDArray[np.float64] = field(repr=False)
    marker: float
    tangent: NDArray[np.float64] = field(repr=False)
    residual_norm: float
    constraint_residual: float
    correction_count: int
    correction_norm: float

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float).copy()
        tangent = np.asarray(self.tangent, dtype=float).copy()
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError("path state must be a finite vector")
        if tangent.shape != (state.size + 1,) or not np.all(np.isfinite(tangent)):
            raise ValueError("path tangent shape is invalid")
        if not math.isfinite(self.marker):
            raise ValueError("path marker must be finite")
        if (not math.isfinite(self.residual_norm) or self.residual_norm < 0.0
                or not math.isfinite(self.constraint_residual)
                or not math.isfinite(self.correction_norm) or self.correction_norm < 0.0
                or type(self.correction_count) is not int or self.correction_count < 0):
            raise ValueError("path acceptance diagnostics are invalid")
        state.setflags(write=False)
        tangent.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "tangent", tangent)


@dataclass(frozen=True, slots=True)
class LocatedPathEvent:
    lower: PathState
    critical: PathState
    upper: PathState


def _normalized(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise PathContinuationError("path direction is zero or non-finite")
    return np.asarray(vector / norm, dtype=float)


def _validated_system(
    state: NDArray[np.float64],
    marker: float,
    residual: ResidualFunction,
    jacobian: JacobianFunction,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    value = np.asarray(residual(state, marker), dtype=float)
    stiffness, marker_derivative = jacobian(state, marker)
    stiffness = np.asarray(stiffness, dtype=float)
    marker_derivative = np.asarray(marker_derivative, dtype=float)
    size = state.size
    if value.shape != (size,):
        raise PathContinuationError("path residual shape is invalid")
    if stiffness.shape != (size, size):
        raise PathContinuationError("path state Jacobian shape is invalid")
    if marker_derivative.shape != (size,):
        raise PathContinuationError("path marker derivative shape is invalid")
    if not all(
        np.all(np.isfinite(item)) for item in (value, stiffness, marker_derivative)
    ):
        raise PathContinuationError("path system contains a non-finite value")
    return value, stiffness, marker_derivative


def initialize_path_state(
    *,
    state: object,
    marker: object,
    next_state: object,
    next_marker: object,
) -> PathState:
    """Create the first committed path state from two accepted equilibria."""

    previous = np.asarray(state, dtype=float)
    current = np.asarray(next_state, dtype=float)
    if previous.ndim != 1 or current.shape != previous.shape:
        raise ValueError("initial path states must be equal-length vectors")
    lower_marker = float(marker)
    current_marker = float(next_marker)
    direction = _normalized(
        np.concatenate((current - previous, [current_marker - lower_marker]))
    )
    return PathState(
        state=current,
        marker=current_marker,
        tangent=direction,
        residual_norm=0.0,
        constraint_residual=0.0,
        correction_count=0,
        correction_norm=0.0,
    )


def initialize_equilibrium_path_state(
    *,
    state: object,
    marker: object,
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    marker_direction: object,
    residual_tolerance: object,
    reference_direction: object | None = None,
) -> PathState:
    """Initialize a path direction from one accepted equilibrium."""

    current = np.asarray(state, dtype=float)
    current_marker = float(marker)
    direction = float(marker_direction)
    tolerance = float(residual_tolerance)
    if current.ndim != 1 or not np.all(np.isfinite(current)):
        raise ValueError("initial path state must be a finite vector")
    if not math.isfinite(current_marker):
        raise ValueError("marker must be finite")
    if not math.isfinite(direction) or direction == 0.0:
        raise ValueError("marker_direction must be finite and nonzero")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    value, stiffness, marker_derivative = _validated_system(
        current, current_marker, residual, jacobian
    )
    residual_norm = float(np.linalg.norm(value))
    if residual_norm > tolerance:
        raise PathContinuationError("initial path state is not an equilibrium")
    reference = np.zeros(current.size + 1, dtype=float)
    reference[-1] = math.copysign(1.0, direction)
    if reference_direction is not None:
        reference=np.asarray(reference_direction,dtype=float)
        if reference.shape!=(current.size+1,) or not np.all(np.isfinite(reference)) or not np.any(reference):
            raise ValueError('invalid accepted reference direction')
    tangent = _oriented_equilibrium_tangent(
        stiffness, marker_derivative, reference
    )
    return PathState(
        state=current,
        marker=current_marker,
        tangent=tangent,
        residual_norm=residual_norm,
        constraint_residual=0.0,
        correction_count=0,
        correction_norm=0.0,
    )


def _oriented_equilibrium_tangent(
    stiffness: NDArray[np.float64],
    marker_derivative: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    equilibrium_jacobian = np.column_stack((stiffness, marker_derivative))
    try:
        _, singular, right = np.linalg.svd(equilibrium_jacobian, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise PathContinuationError("path tangent factorization failed") from exc
    tolerance=max(equilibrium_jacobian.shape)*np.finfo(float).eps*singular[0]
    if np.count_nonzero(singular>tolerance)!=equilibrium_jacobian.shape[0]:
        raise PathContinuationError("path equilibrium Jacobian rank has extra null modes")
    tangent = _normalized(np.asarray(right[-1], dtype=float))
    if float(np.dot(tangent, reference)) < 0.0:
        tangent = -tangent
    full=np.vstack((equilibrium_jacobian,tangent))
    if np.linalg.matrix_rank(full)!=full.shape[0]:
        raise PathContinuationError("full augmented path Jacobian rank is deficient")
    return tangent


def _correct_to_hyperplane(
    *,
    predicted: NDArray[np.float64],
    normal: NDArray[np.float64],
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    settings: PseudoArcLengthSettings,
    reference_tangent: NDArray[np.float64],
) -> PathState:
    combined = np.asarray(predicted, dtype=float).copy()
    normal = _normalized(np.asarray(normal, dtype=float))
    if combined.ndim != 1 or combined.size < 2 or normal.shape != combined.shape:
        raise PathContinuationError("path predictor shape is invalid")
    correction_count = 0
    constraint = math.inf
    residual_norm = math.inf
    correction_norm = math.inf
    for correction_count in range(settings.maximum_corrections + 1):
        state = combined[:-1]
        marker = float(combined[-1])
        value, stiffness, marker_derivative = _validated_system(
            state, marker, residual, jacobian
        )
        constraint = float(np.dot(combined - predicted, normal))
        residual_norm = float(np.linalg.norm(value))
        if (
            residual_norm <= settings.residual_tolerance
            and abs(constraint) <= settings.constraint_tolerance
            and math.isfinite(correction_norm)
            and correction_norm <= settings.correction_tolerance
        ):
            tangent = _oriented_equilibrium_tangent(
                stiffness, marker_derivative, reference_tangent
            )
            return PathState(
                state=state,
                marker=marker,
                tangent=tangent,
                residual_norm=residual_norm,
                constraint_residual=constraint,
                correction_count=correction_count,
                correction_norm=correction_norm,
            )
        if correction_count >= settings.maximum_corrections:
            break
        augmented = np.block(
            [
                [stiffness, marker_derivative[:, None]],
                [normal[None, :-1], normal[None, -1:]],
            ]
        )
        right_hand_side = -np.concatenate((value, [constraint]))
        try:
            correction = np.linalg.solve(augmented, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise PathContinuationError("path correction system is singular") from exc
        if not np.all(np.isfinite(correction)):
            raise PathContinuationError("path correction is non-finite")
        combined += correction
        correction_norm = float(np.linalg.norm(correction[:-1]))
    error=PathContinuationError(
        "path correction did not converge within maximum_corrections"
    )
    frozen=combined.copy();frozen.setflags(write=False)
    predicted_copy=predicted.copy();predicted_copy.setflags(write=False)
    normal_copy=normal.copy();normal_copy.setflags(write=False)
    error.correction_diagnostics=dict(combined=frozen,predicted=predicted_copy,normal=normal_copy,
        residual_norm=residual_norm,constraint_residual=constraint,
        correction_norm=correction_norm,correction_count=correction_count)
    raise error


def pseudo_arclength_step(
    committed: PathState,
    *,
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    settings: PseudoArcLengthSettings,
) -> PathState:
    """Advance one immutable predictor-corrector pseudo-arclength step."""

    current = np.concatenate((committed.state, [committed.marker]))
    predicted = current + settings.arc_length * committed.tangent
    try:
        return _correct_to_hyperplane(
            predicted=predicted,normal=committed.tangent,residual=residual,
            jacobian=jacobian,settings=settings,reference_tangent=committed.tangent)
    except PathContinuationError as error:
        if hasattr(error,'correction_diagnostics'):
            error.correction_diagnostics['last_accepted']=committed
        raise


def locate_path_event(
    lower: PathState,
    upper: PathState,
    *,
    event_function: Callable[[PathState], float],
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    settings: PseudoArcLengthSettings,
    event_tolerance: float,
) -> LocatedPathEvent:
    """Locate an event between two accepted path states without committing trials."""

    tolerance = float(event_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("event_tolerance must be finite and positive")
    lower_value = float(event_function(lower))
    upper_value = float(event_function(upper))
    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
        raise PathContinuationError("path event function is non-finite")
    if lower_value > 0.0 or upper_value < 0.0:
        raise PathContinuationError("path event is not bracketed")
    left = lower
    right = upper
    critical = upper
    for _ in range(80):
        left_combined = np.concatenate((left.state, [left.marker]))
        right_combined = np.concatenate((right.state, [right.marker]))
        chord = right_combined - left_combined
        predicted = 0.5 * (left_combined + right_combined)
        critical = _correct_to_hyperplane(
            predicted=predicted,
            normal=chord,
            residual=residual,
            jacobian=jacobian,
            settings=settings,
            reference_tangent=left.tangent,
        )
        value = float(event_function(critical))
        if not math.isfinite(value):
            raise PathContinuationError("path event function is non-finite")
        if abs(value) <= tolerance:
            return LocatedPathEvent(lower=left, critical=critical, upper=right)
        if value < 0.0:
            left = critical
        else:
            right = critical
    raise PathContinuationError("path event could not be located within tolerance")


def trace_to_path_event(
    committed: PathState,
    *,
    event_function: Callable[[PathState], float],
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    settings: PseudoArcLengthSettings,
    event_tolerance: float,
    maximum_steps: int,
) -> LocatedPathEvent:
    """Trace immutable equilibria until the first requested event is bracketed."""

    if not isinstance(maximum_steps, int) or isinstance(maximum_steps, bool):
        raise ValueError("maximum_steps must be a positive integer")
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be a positive integer")
    lower = committed
    lower_value = float(event_function(lower))
    if not math.isfinite(lower_value):
        raise PathContinuationError("path event function is non-finite")
    if lower_value >= 0.0:
        raise PathContinuationError("initial path state is not below the event")
    for _ in range(maximum_steps):
        upper = pseudo_arclength_step(
            lower,
            residual=residual,
            jacobian=jacobian,
            settings=settings,
        )
        upper_value = float(event_function(upper))
        if not math.isfinite(upper_value):
            raise PathContinuationError("path event function is non-finite")
        if upper_value >= 0.0:
            return locate_path_event(
                lower,
                upper,
                event_function=event_function,
                residual=residual,
                jacobian=jacobian,
                settings=settings,
                event_tolerance=event_tolerance,
            )
        lower = upper
    raise PathContinuationError("path event was not reached within maximum_steps")


__all__ = [
    "LocatedPathEvent",
    "PathContinuationError",
    "PathState",
    "PseudoArcLengthSettings",
    "initialize_equilibrium_path_state",
    "initialize_path_state",
    "locate_path_event",
    "pseudo_arclength_step",
    "trace_to_path_event",
]
