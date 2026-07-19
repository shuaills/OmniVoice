#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.
"""

import logging
import math
import os
import sys
import time
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint
from omnivoice.training.split_loss import (
    SplitLossNumerators,
    WindowCoordinator,
    backward_scalar,
    objective_from_global_sums,
)

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator
        self.accelerator = self._init_accelerator()
        if self.config.split_loss:
            distributed_type = getattr(
                self.accelerator.distributed_type,
                "name",
                str(self.accelerator.distributed_type),
            ).upper()
            if distributed_type in {"DEEPSPEED", "FSDP"}:
                raise RuntimeError(
                    f"split loss is not derived for {distributed_type}; refusing to run"
                )

        # 2. Setup Optimizer & Scheduler if not provided
        if optimizer is None:
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        # 3. DeepSpeed Hack (Batch Size fix)
        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        # 4. Prepare with Accelerator
        (self.model, self.optimizer, self.lr_scheduler,) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        self.global_step = 0
        self.epoch = 0

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging."""
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        # Init handlers
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        # DeepSpeed setup
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            if not os.path.exists(self.config.deepspeed_config):
                raise FileNotFoundError(
                    f"DeepSpeed config not found: {self.config.deepspeed_config}"
                )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with="tensorboard",
            project_dir=self.config.output_dir,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            split_batches=False,
        )

        # Logging setup
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Try to save config if it has the method
            if hasattr(self.config, "save_to_json"):
                self.config.save_to_json(
                    os.path.join(self.config.output_dir, "initial_config.json")
                )

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(self.config.output_dir, "train.log")
                    ),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info(f"Loaded Config: {self.config}")
        set_seed(self.config.seed)
        accelerator.init_trackers("tensorboard")
        return accelerator

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler."""
        if getattr(self.model, "_block_anchor_freeze_base", False):
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise RuntimeError(
                    "anchor-only training has no parameters with "
                    "requires_grad=True"
                )
        else:
            # Preserve the historical optimizer/checkpoint parameter ordering
            # for every existing training mode.  Filtering frozen parameters
            # is an explicit contract only for the anchor mechanism proof.
            trainable_parameters = list(self.model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=getattr(self.config, "perf_fused_adamw", False) or None,
        )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler

    def save_checkpoint(self, step):
        """Wrapper for engine save_checkpoint."""
        engine_save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )
        # Save config copy for convenience
        if self.accelerator.is_main_process and hasattr(self.config, "save_to_json"):
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return {}
        if self.config.split_loss:
            raise RuntimeError(
                "split-loss evaluation requires globally normalized eval windows; "
                "legacy local-mean evaluation is intentionally disabled"
            )

        self.model.eval()
        logger.info(f"Running evaluation at step {self.global_step}...")

        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        with torch.no_grad():
            for eval_batch in self.eval_dataloader:
                eval_batch = _to_device(eval_batch, self.accelerator.device)
                outputs = self.model(**eval_batch)
                local_loss_sum += outputs.loss.detach()
                eval_count += 1

        if eval_count > 0:
            local_mean = local_loss_sum / eval_count
        else:
            local_mean = torch.tensor(0.0, device=self.accelerator.device)

        all_means = self.accelerator.gather(local_mean)
        final_eval_loss = all_means.mean().item()

        eval_metrics = {"eval/loss": final_eval_loss}
        self.accelerator.log(eval_metrics, step=self.global_step)
        logger.info(f"Eval Loss: {final_eval_loss:.4f}")

        self.accelerator.wait_for_everyone()
        self.model.train()
        return eval_metrics

    def train(self):
        """Main training loop."""
        if self.config.split_loss:
            return self._train_split()
        logger.info("Starting Training Loop...")

        # Resume if configured
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
            if getattr(self.config, "force_lr_from_config_on_resume", False):
                base = self.config.learning_rate
                inner = self.lr_scheduler
                while hasattr(inner, "scheduler"):
                    inner = inner.scheduler
                inner.base_lrs = [base] * len(inner.base_lrs)
                lam = inner.lr_lambdas[0](inner.last_epoch)
                for group in self.optimizer.param_groups:
                    group["initial_lr"] = base
                    group["lr"] = base * lam
                logger.info(
                    f"force_lr_from_config_on_resume: base_lr={base}, "
                    f"lr at resumed step {inner.last_epoch} = {base * lam:.3e}"
                )

        # Handle IterableDataset Epochs
        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        # Logger
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)

        self.model.train()
        train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0

        while self.global_step < self.config.steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)

                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            batch = _to_device(batch, self.accelerator.device)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                tr_loss += loss.detach()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_norm = (
                            grad_norm.item() if grad_norm is not None else 0.0
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    train_logger.update(
                        step=self.global_step, loss=loss.item(), lr=current_lr
                    )

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )

                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        current_interval_loss = tr_loss_scalar - logging_loss_scalar
                        avg_loss = current_interval_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar

                        logs = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": self.epoch,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        train_logger.log_metrics(step=self.global_step, metrics=logs)

                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    # Evaluate
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        self.evaluate()

                    # Save
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

        # Final Save
        self.save_checkpoint(self.global_step)
        train_logger.close()
        self.accelerator.end_training()

    def _reduce_split_numerators(
        self, numerators: SplitLossNumerators
    ) -> SplitLossNumerators:
        vector = torch.cat(
            (
                numerators.audio_sum.detach().reshape(-1),
                numerators.eos_sum.detach().reshape(1),
                numerators.void_event_sum.detach().reshape(1),
            )
        ).to(device=self.accelerator.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        num_codebooks = numerators.audio_sum.numel()
        return SplitLossNumerators(
            audio_sum=vector[:num_codebooks],
            eos_sum=vector[num_codebooks],
            void_event_sum=vector[num_codebooks + 1],
        )

    def _train_split(self):
        """Train with globally normalized acoustic/EOS/void loss windows."""

        logger.info("Starting split-loss Training Loop...")
        world_size = self.accelerator.num_processes
        ga_steps = self.accelerator.gradient_accumulation_steps
        if ga_steps != self.config.gradient_accumulation_steps:
            raise RuntimeError(
                "Accelerate/config gradient accumulation mismatch: "
                f"{ga_steps} != {self.config.gradient_accumulation_steps}"
            )
        logger.info(
            "Split-loss runtime: torch=%s accelerate=%s W=%d G=%d",
            torch.__version__,
            accelerate.__version__,
            world_size,
            ga_steps,
        )

        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
            if getattr(self.config, "force_lr_from_config_on_resume", False):
                base = self.config.learning_rate
                inner = self.lr_scheduler
                while hasattr(inner, "scheduler"):
                    inner = inner.scheduler
                inner.base_lrs = [base] * len(inner.base_lrs)
                lam = inner.lr_lambdas[0](inner.last_epoch)
                for group in self.optimizer.param_groups:
                    group["initial_lr"] = base
                    group["lr"] = base * lam
                logger.info(
                    "force_lr_from_config_on_resume: base_lr=%s, "
                    "lr at resumed step %s = %.3e",
                    base,
                    inner.last_epoch,
                    base * lam,
                )

        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)
        coordinator = WindowCoordinator(
            self.train_dataloader,
            ga_steps,
            epoch=self.epoch,
            collective_device=self.accelerator.device,
        )
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)
        self.model.train()

        normalized_weights = torch.tensor(
            self.accelerator.unwrap_model(
                self.model
            ).normalized_audio_codebook_weights,
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        logging_start_time = time.time()
        logging_start_step = self.global_step
        interval = {"loss": 0.0, "audio": 0.0, "eos": 0.0, "void": 0.0}
        interval_audio_counts = [0.0] * normalized_weights.numel()
        interval_void_counts = [0.0] * normalized_weights.numel()
        interval_eos_events = 0.0
        interval_void_events = 0.0
        interval_void_displaced = 0.0

        while self.global_step < self.config.steps:
            previous_epoch = self.epoch
            window = coordinator.next_window()
            self.epoch = window.epoch
            if self.epoch != previous_epoch:
                logger.info("Epoch %d starting. Resetting dataloader...", self.epoch)

            local_audio_sum = None
            local_eos_sum = None
            local_void_sum = None
            grad_norm = 0.0

            for microbatch_index, cpu_batch in enumerate(window.batches):
                batch = _to_device(cpu_batch, self.accelerator.device)
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    if any(
                        value is None
                        for value in (
                            outputs.audio_sum,
                            outputs.eos_sum,
                            outputs.void_event_sum,
                            outputs.audio_count,
                            outputs.eos_count,
                            outputs.void_count,
                            outputs.void_events,
                        )
                    ):
                        raise RuntimeError(
                            "model did not return complete split-loss outputs"
                        )
                    numerators = SplitLossNumerators(
                        audio_sum=outputs.audio_sum,
                        eos_sum=outputs.eos_sum,
                        void_event_sum=outputs.void_event_sum,
                    )
                    loss_for_backward = backward_scalar(
                        numerators,
                        window.global_counts,
                        normalized_weights,
                        gamma=self.config.split_gamma,
                        lambda_eos=self.config.lambda_eos,
                        lambda_void=self.config.lambda_void,
                        world_size=world_size,
                        gradient_accumulation_steps=ga_steps,
                    )

                    expected_sync = microbatch_index == ga_steps - 1
                    if self.accelerator.sync_gradients != expected_sync:
                        raise RuntimeError(
                            "Accelerate sync_gradients violated the exact GA window: "
                            f"microbatch={microbatch_index}, G={ga_steps}, "
                            f"sync_gradients={self.accelerator.sync_gradients}"
                        )
                    self.accelerator.backward(loss_for_backward)

                    detached_audio = outputs.audio_sum.detach().to(torch.float64)
                    detached_eos = outputs.eos_sum.detach().to(torch.float64)
                    detached_void = outputs.void_event_sum.detach().to(torch.float64)
                    if local_audio_sum is None:
                        local_audio_sum = torch.zeros_like(detached_audio)
                        local_eos_sum = torch.zeros_like(detached_eos)
                        local_void_sum = torch.zeros_like(detached_void)
                    local_audio_sum += detached_audio
                    local_eos_sum += detached_eos
                    local_void_sum += detached_void

                    if expected_sync:
                        if self.config.max_grad_norm > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                            grad_norm = (
                                grad_norm.item() if grad_norm is not None else 0.0
                            )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1

            global_numerators = self._reduce_split_numerators(
                SplitLossNumerators(
                    audio_sum=local_audio_sum,
                    eos_sum=local_eos_sum,
                    void_event_sum=local_void_sum,
                )
            )
            global_loss, audio_loss, eos_loss, void_loss = objective_from_global_sums(
                global_numerators,
                window.global_counts,
                normalized_weights,
                gamma=self.config.split_gamma,
                lambda_eos=self.config.lambda_eos,
                lambda_void=self.config.lambda_void,
            )
            metrics = {
                "loss": float(global_loss.item()),
                "audio": float(audio_loss.item()),
                "eos": float(eos_loss.item()),
                "void": float(void_loss.item()),
            }
            for key, value in metrics.items():
                interval[key] += value
            for codebook, count in enumerate(window.global_counts.audio_count.tolist()):
                interval_audio_counts[codebook] += float(count)
            for codebook, count in enumerate(window.global_counts.void_count.tolist()):
                interval_void_counts[codebook] += float(count)
            interval_eos_events += float(window.global_counts.eos_count.item())
            interval_void_events += float(window.global_counts.void_events.item())
            interval_void_displaced += float(
                window.global_counts.void_displaced.item()
            )

            current_lr = self.lr_scheduler.get_last_lr()[0]
            train_logger.update(
                step=self.global_step, loss=metrics["loss"], lr=current_lr
            )
            if self.global_step % self.config.logging_steps == 0:
                elapsed = time.time() - logging_start_time
                completed = self.global_step - logging_start_step
                divisor = float(completed)
                logs = {
                    "train/loss": interval["loss"] / divisor,
                    "train/audio_loss": interval["audio"] / divisor,
                    "train/eos_loss": interval["eos"] / divisor,
                    "train/void_loss": interval["void"] / divisor,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm,
                    "train/epoch": self.epoch,
                    "train/steps_per_sec": completed / elapsed if elapsed > 0 else 0.0,
                    "train/eos_events": interval_eos_events / divisor,
                    "train/void_events": interval_void_events / divisor,
                    "train/void_cells_displaced": (
                        interval_void_displaced / divisor
                    ),
                }
                for codebook in range(normalized_weights.numel()):
                    logs[f"train/audio_cells_c{codebook}"] = (
                        interval_audio_counts[codebook] / divisor
                    )
                    logs[f"train/void_cells_c{codebook}"] = (
                        interval_void_counts[codebook] / divisor
                    )
                train_logger.log_metrics(step=self.global_step, metrics=logs)
                interval = {"loss": 0.0, "audio": 0.0, "eos": 0.0, "void": 0.0}
                interval_audio_counts = [0.0] * normalized_weights.numel()
                interval_void_counts = [0.0] * normalized_weights.numel()
                interval_eos_events = 0.0
                interval_void_events = 0.0
                interval_void_displaced = 0.0
                logging_start_time = time.time()
                logging_start_step = self.global_step

            if (
                self.eval_dataloader is not None
                and self.global_step % self.config.eval_steps == 0
            ):
                self.evaluate()
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(self.global_step)

        self.save_checkpoint(self.global_step)
        train_logger.close()
        self.accelerator.end_training()
