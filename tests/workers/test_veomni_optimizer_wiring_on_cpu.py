# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

pytest.importorskip("veomni")

from verl.workers.config import VeOmniOptimizerConfig
from verl.workers.engine.veomni import transformer_impl as veomni_impl


def _make_engine(optimizer_config):
    engine = object.__new__(veomni_impl.VeOmniEngine)
    engine.optimizer_config = optimizer_config
    engine.data_parallel_mode = "fsdp2"
    return engine


def test_optimizer_override_config_is_forwarded(monkeypatch):
    captured = {}
    expected_optimizer = object()

    def fake_build_optimizer(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return expected_optimizer

    monkeypatch.setattr(veomni_impl, "build_optimizer", fake_build_optimizer)
    config = VeOmniOptimizerConfig(
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        override_optimizer_config={"eps": 1e-6, "fused": True, "betas": (0.8, 0.9)},
    )
    engine = _make_engine(config)
    model = object()

    optimizer = engine._build_optimizer(model)

    assert optimizer is expected_optimizer
    assert captured == {
        "model": model,
        "lr": 2e-4,
        "betas": (0.8, 0.9),
        "weight_decay": 0.1,
        "optimizer_type": "adamw",
        "eps": 1e-6,
        "fused": True,
    }


@pytest.mark.parametrize(
    ("lr_warmup_steps", "lr_warmup_steps_ratio", "expected_ratio"),
    [
        (20, 0.1, 0.04),
        (-1, 0.1, 0.1),
    ],
)
def test_lr_scheduler_honors_absolute_warmup_steps(monkeypatch, lr_warmup_steps, lr_warmup_steps_ratio, expected_ratio):
    captured = {}
    expected_scheduler = object()

    def fake_build_lr_scheduler(optimizer, **kwargs):
        captured["optimizer"] = optimizer
        captured.update(kwargs)
        return expected_scheduler

    monkeypatch.setattr(veomni_impl, "build_lr_scheduler", fake_build_lr_scheduler)
    config = VeOmniOptimizerConfig(
        total_training_steps=500,
        lr_warmup_steps=lr_warmup_steps,
        lr_warmup_steps_ratio=lr_warmup_steps_ratio,
    )
    engine = _make_engine(config)
    optimizer = object()

    scheduler = engine._build_lr_scheduler(optimizer)

    assert scheduler is expected_scheduler
    assert captured["lr_warmup_ratio"] == pytest.approx(expected_ratio)


def test_absolute_warmup_steps_require_positive_total_steps(monkeypatch):
    monkeypatch.setattr(veomni_impl, "build_lr_scheduler", lambda *args, **kwargs: object())
    config = VeOmniOptimizerConfig(total_training_steps=-1, lr_warmup_steps=10)
    engine = _make_engine(config)

    with pytest.raises(ValueError, match="total_training_steps must be positive"):
        engine._build_lr_scheduler(object())
