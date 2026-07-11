from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from apps.streamlit.result_service import (
    DiscoveryLimits,
    artifact_groups,
    discover_results,
    headline_metrics,
    load_metric_table,
)
from apps.streamlit.run_service import RunRecord


def record(tmp_path: Path, *, output_root: str = 'results/demo') -> tuple[Path, RunRecord]:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'main.py').write_text('')
    (repo / 'configs').mkdir()
    run_dir = repo / 'outputs' / 'streamlit' / 'run-1'
    run_dir.mkdir(parents=True)
    rec = RunRecord(
        run_id='run-1',
        status='succeeded',
        run_dir=run_dir,
        command=('python', 'main.py'),
        output_root=output_root,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    return repo, rec


def test_discovers_metrics_images_and_logs(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = repo / 'results' / 'demo' / 'exp' / 'iter_0'
    result_dir.mkdir(parents=True)
    (result_dir / 'all_results.csv').write_text('acc,loss\n0.9,0.1\n', encoding='utf-8')
    (result_dir / 'plot.png').write_bytes(b'png')
    (rec.run_dir / 'run.log').write_text('done', encoding='utf-8')
    bundle = discover_results(repo, rec)
    groups = artifact_groups(bundle)
    assert groups['image'][0].path.name == 'plot.png'
    assert groups['log'][0].path.name == 'run.log'
    assert bundle.metrics[0].rows[0]['acc'] == '0.9'
    assert headline_metrics(bundle.metrics)[0] == ('acc', 0.9)


def test_malformed_json_becomes_warning(tmp_path: Path):
    path = tmp_path / 'metrics.json'
    path.write_text('{bad', encoding='utf-8')
    table = load_metric_table(path)
    assert not table.rows
    assert 'Could not parse metrics' in table.warning


def test_large_metric_file_is_not_parsed(tmp_path: Path):
    path = tmp_path / 'all_results.csv'
    path.write_text('a\n' + ('1\n' * 100), encoding='utf-8')
    table = load_metric_table(path, DiscoveryLimits(max_metric_bytes=10))
    assert 'parsing is limited' in table.warning


def test_broad_output_root_is_refused(tmp_path: Path):
    repo, rec = record(tmp_path, output_root='.')
    bundle = discover_results(repo, rec)
    assert any('Refusing to scan' in item for item in bundle.warnings)
    assert bundle.roots == (rec.run_dir.resolve(),)


def test_symlink_escape_is_skipped(tmp_path: Path):
    repo, rec = record(tmp_path)
    output = repo / 'results' / 'demo'
    output.mkdir(parents=True)
    outside = tmp_path / 'outside.json'
    outside.write_text('{"secret": 1}')
    try:
        (output / 'metrics.json').symlink_to(outside)
    except (OSError, NotImplementedError):
        return
    bundle = discover_results(repo, rec)
    assert all(item.path.name != 'metrics.json' for item in bundle.artifacts)


def test_scan_limits_report_truncation(tmp_path: Path):
    repo, rec = record(tmp_path)
    output = repo / 'results' / 'demo'
    output.mkdir(parents=True)
    for index in range(8):
        (output / f'{index}.txt').write_text('x')
    bundle = discover_results(repo, rec, limits=DiscoveryLimits(max_files=3))
    assert bundle.truncated
    assert len(bundle.artifacts) <= 4  # run-dir manifest/log plus bounded output files


def test_headline_metrics_skips_non_finite_values(tmp_path: Path):
    path = tmp_path / 'metrics.json'
    path.write_text('{"loss": "nan", "acc": 0.8}', encoding='utf-8')
    table = load_metric_table(path)
    assert headline_metrics((table,)) == (('acc', 0.8),)
