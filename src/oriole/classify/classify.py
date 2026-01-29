from __future__ import annotations

import os
from dataclasses import dataclass

from ..check import check_params
from ..data import GwasData, load_data, Meta
from ..error import new_error
from ..options.action import Action
from ..options.config import ClassifyConfig, Config
from ..params import Params, read_params_from_file
from ..sample.var_stats import SampledClassification
from ..util.threads import Threads, TaskQueueObserver
from .worker import Classification, MessageToCentral, MessageToWorker, ClassifyWorkerLauncher


class Observer(TaskQueueObserver):
    def __init__(self, var_ids: list[str], out_file: str, meta: Meta) -> None:
        self.meta = meta
        self.var_ids = var_ids
        self.out_file = out_file
        self._writer = open(out_file, "w", encoding="utf-8")

    def going_to_start_queue(self) -> None:
        print("Starting to classify data points.")
        self._writer.write(self._header())

    def going_to_send(self, out_message, i_task: int, i_thread: int) -> None:
        if out_message.kind == "shutdown":
            print(f"Sent shutdown as task {i_task} to thread {i_thread}")

    def have_received(self, in_message: MessageToCentral, i_task: int, i_thread: int) -> None:
        var_id = self.var_ids[i_task]
        try:
            self._writer.write(format_entry(var_id, in_message.classification))
        except Exception as exc:
            print(f"Cannot write temp file: {exc}")

    def nothing_more_to_send(self) -> None:
        print("No more data points to add to queue.")

    def completed_queue(self) -> None:
        print("Completed classification of all data points.")
        self._writer.close()

    def _header(self) -> str:
        traits_part = "\t".join(self.meta.trait_names)
        return f"id\te_mean_samp\te_std_samp\te_mean_calc\t{traits_part}\n"


def classify_or_check(config: Config, dry: bool, analytical: bool = False) -> None:
    params = read_params_from_file(config.files.params)
    check_params(config, params)
    print(f"Read from file mu = {params.mu}, tau = {params.tau}")
    if config.classify.params_override is not None:
        params = params.plus_overwrite(config.classify.params_override)
        print(f"After overwrite, mu = {params.mu}, tau = {params.tau}")
    data = load_data(config, Action.CLASSIFY)
    if dry:
        print("User picked dry run only, so doing nothing.")
        return
    classify(data.gwas_data, params, config, analytical=analytical)


def classify(data: GwasData, params: Params, config: Config, analytical: bool = False) -> None:
    n_threads = max(os.cpu_count() or 1, 3)
    classify_config = config.classify
    launcher = ClassifyWorkerLauncher(data, params, classify_config, analytical=analytical)
    threads = Threads.new(launcher, n_threads)
    meta = data.meta
    out_messages = (MessageToWorker.data_point(i) for i in range(meta.n_data_points()))
    temp_out_file = f"{classify_config.out_file}_tmp"
    observer = Observer(meta.var_ids, temp_out_file, meta)
    in_messages = threads.task_queue(out_messages, observer)
    classifications = [message.classification for message in in_messages]
    write_out_file(classify_config.out_file, meta, classifications)
    threads.close(MessageToWorker.shutdown())


def write_out_file(file: str, meta: Meta, classifications: list[Classification]) -> None:
    with open(file, "w", encoding="utf-8") as handle:
        handle.write(format_header(meta))
        for var_id, classification in zip(meta.var_ids, classifications):
            handle.write(format_entry(var_id, classification))


def format_header(meta: Meta) -> str:
    traits_part = "\t".join(meta.trait_names)
    return f"id\te_mean_samp\te_std_samp\te_mean_calc\t{traits_part}\n"


def format_entry(var_id: str, classification: Classification) -> str:
    sampled: SampledClassification = classification.sampled
    t_means_part = "\t".join(str(value) for value in sampled.t_means)
    return (
        f"{var_id}\t{sampled.e_mean}\t{sampled.e_std}\t"
        f"{classification.e_mean_calculated}\t{t_means_part}\n"
    )
