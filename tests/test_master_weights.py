"""fp32 master weights: openrl.master_dtype is validated, recorded, carried to
the trainer and sized for, and the FFT trainer's MasterWeightAdamW trains bf16
parameters through fp32 masters."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from server import api_server
from server.estimator import footprint
from server.model_metadata import TrainingModelMetadata
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import worker_env
from tests.test_session_lifecycle import RuntimeManager
from training import fft_trainer_worker
from training.fft_trainer_worker import FFTTrainingWorker
from training.master_weights import MasterWeightAdamW
from training.trainer_worker import BaseTrainerWorker
from training.types import check_master_dtype


def dtypes(trainer: str = "bfloat16", master: str | None = None) -> dict[str, str]:
  out = {"openrl.trainer_dtype": trainer, "openrl.sampler_dtype": trainer}
  if master is not None:
    out["openrl.master_dtype"] = master
  return out


class MasterWeightAdamWTest(unittest.TestCase):
  HYPER = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}

  def params(self, dtype: torch.dtype) -> list[torch.nn.Parameter]:
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(8, 4).to(dtype)), torch.nn.Parameter(torch.randn(5).to(dtype))]

  def run_steps(self, optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter], steps: int) -> None:
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
      for param in params:
        param.grad = torch.randn(param.shape, generator=generator).to(param.dtype)
      optimizer.step()
      optimizer.zero_grad()

  def test_matches_torch_adamw_on_fp32_parameters(self) -> None:
    ours, reference = self.params(torch.float32), self.params(torch.float32)
    self.run_steps(MasterWeightAdamW(ours, **self.HYPER), ours, 10)
    self.run_steps(torch.optim.AdamW(reference, **self.HYPER), reference, 10)
    for mine, theirs in zip(ours, reference, strict=True):
      torch.testing.assert_close(mine, theirs, rtol=1e-6, atol=1e-7)

  def test_bf16_parameters_follow_the_fp32_masters(self) -> None:
    # A constant gradient makes each Adam step move a weight by about lr. At
    # lr=1e-4 one step is below half a bf16 ulp for any weight above ~0.05, so
    # pure bf16 AdamW leaves most weights where they started; 200 steps add up
    # to ~2e-2 in the fp32 masters, which moves nearly all of them.
    def moved_fraction(optimizer_class, params: list[torch.nn.Parameter]) -> tuple[float, torch.optim.Optimizer]:
      start = [param.detach().clone() for param in params]
      optimizer = optimizer_class(params, lr=1e-4, weight_decay=0.0)
      for _ in range(200):
        for param in params:
          param.grad = torch.ones_like(param)
        optimizer.step()
      moved = sum(int((param.detach() != before).sum()) for param, before in zip(params, start, strict=True))
      return moved / sum(before.numel() for before in start), optimizer

    params = self.params(torch.bfloat16)
    with_masters, optimizer = moved_fraction(MasterWeightAdamW, params)
    pure_bf16, _ = moved_fraction(torch.optim.AdamW, self.params(torch.bfloat16))
    self.assertGreater(with_masters, 0.9)
    self.assertLess(pure_bf16, 0.5)
    for param in params:
      state = optimizer.state[param]
      self.assertEqual(param.dtype, torch.bfloat16)
      self.assertEqual({state[key].dtype for key in ("master", "exp_avg", "exp_avg_sq")}, {torch.float32})
      torch.testing.assert_close(param.detach(), state["master"].to(torch.bfloat16), rtol=0, atol=0)

  def test_state_dict_round_trip_keeps_the_masters(self) -> None:
    params = self.params(torch.bfloat16)
    optimizer = MasterWeightAdamW(params, lr=1e-3)
    self.run_steps(optimizer, params, 3)
    restored = MasterWeightAdamW(params, lr=1e-3)
    restored.load_state_dict(optimizer.state_dict())
    for param in params:
      torch.testing.assert_close(restored.state[param]["master"], optimizer.state[param]["master"], rtol=0, atol=0)
      self.assertEqual(restored.state[param]["master"].dtype, torch.float32)


class FFTOptimizerChoiceTest(unittest.TestCase):
  def worker(self, trainer: str, master: str) -> FFTTrainingWorker:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_DTYPE": trainer, "OPEN_RL_MASTER_DTYPE": master}):
      worker = FFTTrainingWorker()
    worker.trainable_params = [torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16 if trainer == "bfloat16" else torch.float32))]
    return worker

  def test_fp32_masters_under_bf16_compute_use_the_wrapper(self) -> None:
    self.assertIsInstance(self.worker("bfloat16", "float32").new_optimizer(lr=1e-4), MasterWeightAdamW)

  def test_matching_dtypes_use_plain_adamw(self) -> None:
    for dtype in ("bfloat16", "float32"):
      with self.subTest(dtype=dtype):
        optimizer = self.worker(dtype, dtype).new_optimizer(lr=1e-4)
        self.assertIs(type(optimizer), torch.optim.AdamW)

  def test_the_delta_carries_the_bf16_values_of_the_masters(self) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from training.types import Datum, TensorData

    with patch.dict(os.environ, {"OPEN_RL_TRAINER_DTYPE": "bfloat16", "OPEN_RL_MASTER_DTYPE": "float32"}):
      worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    worker.cpu_offload = False
    worker.weight_sync_cfg.strategy = "delta"
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    worker.model = LlamaForCausalLM(config).to(torch.bfloat16)
    with patch.object(fft_trainer_worker, "ENABLE_GRADIENT_CHECKPOINTING", False):
      worker.prepare_model_for_training()

    targets = {"target_tokens": TensorData(data=[5, 8, 13, 21]), "weights": TensorData(data=[1.0] * 4)}
    data = [Datum(model_input=[3, 5, 8, 13], loss_fn_inputs=targets)]
    for _ in range(3):
      worker.forward_backward(data, "cross_entropy")
      worker.optim_step({"learning_rate": 1e-3})

    self.assertIsInstance(worker.optimizer, MasterWeightAdamW)
    params = dict(worker.model.named_parameters())
    self.assertEqual({param.dtype for param in params.values()}, {torch.bfloat16})
    delta = worker._latest_delta_tensors
    self.assertTrue(delta["names"])
    for name, indices, values in zip(delta["names"], delta["indices_list"], delta["values_list"], strict=True):
      param = params[name]
      # What the sampler gets is the bf16 parameter, i.e. the master rounded.
      self.assertEqual(values.dtype, torch.bfloat16)
      torch.testing.assert_close(values, param.detach().view(-1)[indices.long()], rtol=0, atol=0)
      master = worker.optimizer.state[param]["master"]
      torch.testing.assert_close(param.detach(), master.to(torch.bfloat16), rtol=0, atol=0)


class MasterDtypeRulesTest(unittest.TestCase):
  def test_allowed(self) -> None:
    for kind, trainer, master in (
      ("full", "bfloat16", "float32"),
      ("full", "bfloat16", "bfloat16"),
      ("full", "float32", "float32"),
      ("lora", "bfloat16", "float32"),
    ):
      with self.subTest(kind=kind, trainer=trainer, master=master):
        check_master_dtype(kind, trainer, master)

  def test_refused(self) -> None:
    for kind, trainer, master, reason in (
      ("full", "float32", "bfloat16", "float32 or the trainer dtype"),
      ("lora", "bfloat16", "bfloat16", "LoRA adapters train in float32"),
    ):
      with self.subTest(kind=kind, trainer=trainer, master=master), self.assertRaisesRegex(ValueError, reason):
        check_master_dtype(kind, trainer, master)

  def test_a_worker_refuses_a_model_with_another_master_dtype(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_DTYPE": "bfloat16", "OPEN_RL_MASTER_DTYPE": "float32"}):
      worker = BaseTrainerWorker()
    worker.check_command_dtype("bfloat16", "float32")
    worker.check_command_dtype("bfloat16", None)  # an older server
    with self.assertRaisesRegex(ValueError, "keeps its weights in float32"):
      worker.check_command_dtype("bfloat16", "bfloat16")

  def test_a_hand_started_worker_defaults_to_fp32_masters(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_DTYPE": "bfloat16"}):
      os.environ.pop("OPEN_RL_MASTER_DTYPE", None)
      self.assertEqual(BaseTrainerWorker().master_dtype_name, "float32")


class ApiMasterDtypeTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=api_server.state))
    self.enterContext(patch.object(api_server, "worker_manager", RuntimeManager()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}))
    self.client = TestClient(api_server.app)

  def post(self, path: str, body: dict):
    return self.client.post(f"/api/v1/{path}", json=body)

  def create(self, **user_metadata):
    return self.post("create_model", {"base_model": "m", "user_metadata": user_metadata})

  def metadata(self, model_id: str) -> dict:
    return json.loads(api_server.state.get_value_sync(f"open_rl:model_meta:{model_id}"))

  def queued(self) -> list[dict]:
    return [request["payload"] for request in asyncio.run(api_server.store.get_requests())]

  def test_the_default_is_fp32(self) -> None:
    for kind in ("lora", "full"):
      with self.subTest(kind=kind):
        response = self.create(**{"openrl.fine_tuning_type": kind}, **dtypes())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.metadata(response.json()["request_id"])["master_dtype"], "float32")
        self.assertEqual(self.queued()[-1]["master_dtype"], "float32")

  def test_pure_bf16_full_fine_tuning_is_opt_in(self) -> None:
    response = self.create(**{"openrl.fine_tuning_type": "full"}, **dtypes("bfloat16", "bfloat16"))
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(self.metadata(response.json()["request_id"])["master_dtype"], "bfloat16")

  def test_unsupported_master_dtypes_are_a_400(self) -> None:
    for user_metadata, reason in (
      ({"openrl.fine_tuning_type": "full", **dtypes("float32", "bfloat16")}, "float32 or the trainer dtype"),
      ({**dtypes("bfloat16", "bfloat16")}, "LoRA adapters train in float32"),
    ):
      with self.subTest(reason=reason):
        response = self.create(**user_metadata)
        self.assertEqual(response.status_code, 400)
        self.assertIn(reason, response.json()["error"])

  def restore(self, saved: dict, lora: bool = False, **user_metadata):
    with tempfile.TemporaryDirectory() as directory:
      with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump({"base_model": "checkpoint-base", **saved}, f)
      if lora:
        with open(os.path.join(directory, "adapter_config.json"), "w") as f:
          json.dump({"r": 8}, f)
      return self.post("create_model_from_state", {"state_path": directory, "user_metadata": user_metadata})

  def test_restore_must_match_the_checkpoint_master_dtype(self) -> None:
    response = self.restore({"dtype": "bfloat16", "master_dtype": "bfloat16"}, **dtypes())
    self.assertEqual(response.status_code, 400)
    self.assertIn("whose weights were trained in bfloat16", response.json()["error"])
    self.assertIn("master_dtype defaults to float32", response.json()["error"])
    response = self.restore({"dtype": "bfloat16", "master_dtype": "float32"}, **dtypes("bfloat16", "bfloat16"))
    self.assertEqual(response.status_code, 400)

  def test_restore_with_the_checkpoint_master_dtype(self) -> None:
    response = self.restore({"dtype": "bfloat16", "master_dtype": "bfloat16"}, **dtypes("bfloat16", "bfloat16"))
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(self.queued()[-1]["master_dtype"], "bfloat16")

  def test_a_full_checkpoint_from_before_master_dtype_trained_in_its_dtype(self) -> None:
    # Pure bf16, so the fp32 default is refused rather than upcast.
    self.assertEqual(self.restore({"dtype": "bfloat16"}, **dtypes()).status_code, 400)
    self.assertEqual(self.restore({"dtype": "bfloat16"}, **dtypes("bfloat16", "bfloat16")).status_code, 200)

  def test_a_lora_checkpoint_from_before_master_dtype_is_trusted(self) -> None:
    response = self.restore({"dtype": "bfloat16"}, lora=True, **dtypes())
    self.assertEqual(response.status_code, 200, response.text)


class WorkerEnvAndSizingTest(unittest.TestCase):
  def test_the_trainer_gets_the_master_dtype(self) -> None:
    meta = TrainingModelMetadata(
      base_model="Qwen/Qwen3-0.6B", fine_tuning_type="full", trainer_dtype="bfloat16", sampler_dtype="bfloat16", master_dtype="float32"
    )
    self.assertEqual(worker_env(meta, meta.base_model, "r", False, "trainer")["OPEN_RL_MASTER_DTYPE"], "float32")
    self.assertNotIn("OPEN_RL_MASTER_DTYPE", worker_env(meta, meta.base_model, "r", False, "sampler"))

  def test_a_record_from_before_master_dtype_keeps_what_it_ran_with(self) -> None:
    full = TrainingModelMetadata(base_model="m", fine_tuning_type="full", trainer_dtype="bfloat16", sampler_dtype="bfloat16")
    lora = TrainingModelMetadata(base_model="m", fine_tuning_type="lora", trainer_dtype="bfloat16", sampler_dtype="bfloat16")
    self.assertEqual(full.worker_master_dtype(), "bfloat16")
    self.assertEqual(lora.worker_master_dtype(), "float32")

  def test_fp32_masters_size_like_fp32_training_on_the_device(self) -> None:
    model = "Qwen/Qwen3-4B"
    mixed = footprint(model, "full", "trainer", "bfloat16", "float32")
    pure = footprint(model, "full", "trainer", "bfloat16", "bfloat16")
    self.assertEqual(mixed.accelerator_bytes, footprint(model, "full", "trainer", "float32", "float32").accelerator_bytes)
    self.assertGreater(mixed.host_request_bytes, pure.host_request_bytes)
    self.assertEqual(pure, footprint(model, "full", "trainer", "bfloat16"))
    # LoRA adapters are too small to count; the sampler holds no masters.
    self.assertEqual(footprint(model, "lora", "trainer", "bfloat16", "float32"), footprint(model, "lora", "trainer", "bfloat16"))


if __name__ == "__main__":
  unittest.main()
