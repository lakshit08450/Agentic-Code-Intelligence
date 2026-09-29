from mteb.models.models_protocols import CrossEncoderProtocol, EncoderProtocol

from codeintel.stage2.reranker import ExecutionReranker, Stage2Cfg, is_multi_answer, stage2_score

TRIVIAL = "Solve.\n\n-----Examples-----\nInput\n3\n\nOutput\nYES\n"
RICH = "Solve.\n\n-----Examples-----\nInput\n3\n\nOutput\n1 2 3\n"
MULTI = RICH.replace("Solve.", "If there are multiple answers, print any of them.")


def test_fix4_trivial_pass_downweighted():
    cfg = Stage2Cfg(w_pass=1.0, trivial_pass_factor=0.1)
    assert stage2_score(0.5, "PASS", TRIVIAL, cfg) == 0.5 + 0.1
    assert stage2_score(0.5, "PASS", RICH, cfg) == 1.5


def test_fix3_wrong_neutral_on_multi_answer():
    cfg = Stage2Cfg(w_wrong=0.1)
    assert is_multi_answer(MULTI) and not is_multi_answer(RICH)
    assert stage2_score(0.5, "WRONG", MULTI, cfg) == 0.5
    assert stage2_score(0.5, "WRONG", RICH, cfg) == 0.4
    assert stage2_score(0.5, "WRONG", MULTI, Stage2Cfg(w_wrong=0.1, multi_answer_neutral=False)) == 0.4


def test_unknown_and_timeout_neutral():
    cfg = Stage2Cfg(w_pass=1.0, w_wrong=0.1, w_err=0.05)
    assert stage2_score(0.3, "UNKNOWN", RICH, cfg) == 0.3
    assert stage2_score(0.3, "TIMEOUT", RICH, cfg) == 0.3
    assert stage2_score(0.3, "ERROR", RICH, cfg) == 0.25


def test_mteb_routes_reranker_as_cross_encoder():
    # the class, not an instance: only method presence matters for runtime_checkable protocols
    assert hasattr(ExecutionReranker, "predict") and not hasattr(ExecutionReranker, "encode")
    r = ExecutionReranker.__new__(ExecutionReranker)
    r.mteb_model_meta = None
    assert isinstance(r, CrossEncoderProtocol)
    assert not isinstance(r, EncoderProtocol)
