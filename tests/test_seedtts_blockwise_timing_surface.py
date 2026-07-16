import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "tests" / "seedtts_blockwise_gen.py"


def _decode_call(tree: ast.AST) -> ast.Call:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_decode_block_causal"
    ]
    assert len(calls) == 1
    return calls[0]


def test_timing_flag_defaults_off_and_metadata_is_conditional() -> None:
    source = GENERATOR.read_text()
    tree = ast.parse(source)

    timing_args = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--measure-token-decode"
    ]
    assert len(timing_args) == 1
    assert any(
        keyword.arg == "action"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "store_true"
        for keyword in timing_args[0].keywords
    )
    assert "if args.measure_token_decode:" in source
    assert "meta.update({" in source
    assert "'token_decode_seconds': token_decode_seconds" in source
    assert "'token_decode_rtf': token_decode_rtf" in source
    assert "'timing_warmup': timing_warmup_pending" in source


def test_timing_sync_and_clock_wrap_only_core_token_decode() -> None:
    source = GENERATOR.read_text()
    tree = ast.parse(source)
    call = _decode_call(tree)
    call_line = call.lineno

    sync_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
    ]
    perf_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "perf_counter"
    ]
    assert len(sync_lines) == 2
    assert len(perf_lines) == 2
    assert min(sync_lines) < call_line < max(sync_lines)
    assert min(perf_lines) < call_line < max(perf_lines)
    assert source.index("tok.decode(") > source.index("token_decode_seconds")


def test_timing_warmup_is_consumed_only_after_successful_metadata_write() -> None:
    source = GENERATOR.read_text()

    meta_write = source.index("fh.write(json.dumps(meta")
    warmup_clear = source.index("timing_warmup_pending = False")
    assert warmup_clear > meta_write
