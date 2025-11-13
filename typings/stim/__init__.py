from typing import Iterator, List, Union, Any

class DemTarget:
    """Type stub for stim.DemTarget"""
    @property
    def val(self) -> int: ...
    def is_relative_detector_id(self) -> bool: ...

class DemInstruction:
    """Type stub for stim.DemInstruction"""
    @property
    def type(self) -> str: ...
    def targets_copy(self) -> List[DemTarget]: ...

class DetectorErrorModel:
    """Type stub for stim.DetectorErrorModel"""
    def flattened(self) -> Iterator[DemInstruction]: ...

class CompiledMeasurementSampler:
    """Type stub for sampler objects"""
    # sample 메서드가 튜플을 반환할지 배열을 반환할지 상황에 따라 다르므로 
    # Any로 설정하여 Pylance가 unpacking 에러를 내지 않도록 함
    def sample(self, shots: int, bit_packed: bool = False, separate_observables: bool = False) -> Any: ...

class Circuit:
    """Type stub for stim.Circuit"""
    # Missing method added here:
    def compile_sampler(self) -> CompiledMeasurementSampler: ...
    def compile_detector_sampler(self) -> CompiledMeasurementSampler: ...
    
    def detector_error_model(self, decompose_errors: bool = False, flatten_loops: bool = False, ignore_decomposition_failures: bool = False) -> DetectorErrorModel: ...
    def get_detector_coordinates(self, only: Union[List[int], None] = None) -> dict[int, list[float]]: ...
    
    @staticmethod
    def generated(
        code_task: str,
        distance: int,
        rounds: int,
        after_clifford_depolarization: float = 0,
        after_reset_flip_probability: float = 0,
        before_measure_flip_probability: float = 0,
        before_round_data_depolarization: float = 0
    ) -> 'Circuit': ...