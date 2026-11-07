"""Local research artifacts must never alter isolated unit-test predictions."""
import pytest


@pytest.fixture(autouse=True)
def isolate_shared_population_artifact(monkeypatch,tmp_path):
    monkeypatch.setenv('PULSE_POPULATION_MODEL',str(tmp_path/'absent-population-model.json'))
