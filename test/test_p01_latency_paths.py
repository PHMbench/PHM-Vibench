import torch
from experiments.p01.measure_latency import deployed_prediction, direct_prediction


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reference_temperature = torch.tensor(1.)
        self.reference_calls = 0
        self.candidate_calls = 0

    def reference(self, x):
        self.reference_calls += 1
        return x

    def candidate(self, x):
        self.candidate_calls += 1
        return -x


def test_zero_coefficient_uses_only_reference():
    model = Classifier()
    x = torch.tensor([[2., -1.]])
    result = deployed_prediction(model, x, dict(alpha=0., kind='model', temperature=1.))
    torch.testing.assert_close(result, x.log_softmax(-1))
    assert model.reference_calls == 1
    assert model.candidate_calls == 0


def test_plain_direct_classifier_does_not_compute_reference():
    model = Classifier()
    x = torch.tensor([[2., -1.]])
    result = direct_prediction(model, x, dict(kind='model', temperature=1.))
    torch.testing.assert_close(result, (-x).log_softmax(-1))
    assert model.reference_calls == 0
    assert model.candidate_calls == 1


def test_temperature_deployment_computes_reference_once():
    model = Classifier()
    x = torch.tensor([[2., -1.]])
    result = deployed_prediction(model, x, dict(alpha=.3, kind='temperature', temperature=2.))
    expected = (.7 * x.softmax(-1) + .3 * (x/2).softmax(-1)).log()
    torch.testing.assert_close(result, expected)
    assert model.reference_calls == 1
    assert model.candidate_calls == 0
