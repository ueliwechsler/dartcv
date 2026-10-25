import numpy as np

from dartcv.dartboard_dart_pipeline import DartboardDartRecognitionPipeline


def test_dartboard_pipeline_tracks_stage_outputs_and_timings():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    pipeline = DartboardDartRecognitionPipeline()

    result = pipeline.run(image)

    assert result.board_mask is not None
    assert result.dart_mask is not None
    assert len(result.steps) == 7
    assert set(result.timings_ms) == {step.name for step in result.steps}
    assert all(step.elapsed_ms >= 0 for step in result.steps)
    assert result.summary().count("ms") >= 7
    assert set(result.debug_images()) == {step.name for step in result.steps}


def test_pipeline_accepts_custom_stage_registration():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    pipeline = DartboardDartRecognitionPipeline()

    def custom_stage(image, state):
        state["custom"] = np.ones(image.shape[:2], dtype=np.uint8) * 42
        return state["custom"]

    pipeline.register_stage("custom_stage", custom_stage)
    result = pipeline.run(image)

    assert any(step.name == "custom_stage" for step in result.steps)
    assert result.debug_images()["custom_stage"].shape == image.shape[:2]
