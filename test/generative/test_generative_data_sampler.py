from types import SimpleNamespace

import src.data_factory.samplers.Get_sampler as sampler_module


def test_generative_task_uses_standard_factory_sampler(monkeypatch) -> None:
    expected = object()
    calls = []
    monkeypatch.setattr(
        sampler_module,
        "_get_standard_sampler",
        lambda args_data, dataset, mode, task_name: (
            calls.append((args_data, dataset, mode, task_name)) or expected
        ),
    )
    args_data = SimpleNamespace(batch_size=2)
    dataset = SimpleNamespace(
        file_windows_list=[{"file_id": 1}],
        metadata={1: {"Dataset_id": 0}},
    )

    result = sampler_module.Get_sampler(
        SimpleNamespace(type="generative"),
        args_data,
        dataset,
        mode="train",
    )

    assert result is expected
    assert calls == [(args_data, dataset, "train", "Pretrain")]
