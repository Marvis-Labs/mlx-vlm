from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ComponentRegistration:
    name: str
    planner_factory: Callable[["ComponentServices"], tuple[Any, ...]]
    output_factory: Callable[[], Any]
    work: frozenset[tuple[str, str]]
    phases: frozenset[str]


class ComponentServices:
    def __init__(
        self,
        config_directory: Path,
        repository: Path,
        mlp_config: Path | None,
        model_config: Path,
        scenario_config: Path,
    ):
        self.config_directory = config_directory
        self.repository = repository
        self.mlp_config = mlp_config
        self.model_config = model_config
        self.scenario_config = scenario_config
        self._model_path = None

    def model_path(self):
        if self._model_path is None:
            from ci.delegator import ModelPath

            self._model_path = ModelPath(
                self.model_config,
                self.scenario_config,
            )
        return self._model_path


def _model_path_planners(services: ComponentServices) -> tuple[Any, ...]:
    from ci.delegator import NewModelPath

    model_path = services.model_path()
    return NewModelPath(model_path), model_path


def _mlp_planners(services: ComponentServices) -> tuple[Any, ...]:
    if services.mlp_config is None:
        return ()
    from ci.mlp_change import GitSource, MLPChange

    return (
        MLPChange(
            services.mlp_config,
            services.model_path(),
            GitSource(services.repository),
        ),
    )


def _kv_cache_planners(services: ComponentServices) -> tuple[Any, ...]:
    from ci.kv_cache_change import KVCacheChange
    from ci.mlp_change import GitSource

    return (
        KVCacheChange(
            services.config_directory / "components" / "kv_cache_profiles.yaml",
            GitSource(services.repository),
        ),
    )


def _model_path_output():
    from ci.bot import ModelPathOutput

    return ModelPathOutput()


def _mlp_output():
    from ci.bot import MLPChangeOutput

    return MLPChangeOutput()


def _kv_cache_output():
    from ci.bot import KVCacheChangeOutput

    return KVCacheChangeOutput()


REGISTRATIONS = (
    ComponentRegistration(
        name="mlp_change",
        planner_factory=_mlp_planners,
        output_factory=_mlp_output,
        work=frozenset(),
        phases=frozenset({"mlp_contract"}),
    ),
    ComponentRegistration(
        name="kv_cache_change",
        planner_factory=_kv_cache_planners,
        output_factory=_kv_cache_output,
        work=frozenset({("KVCacheChange", "kv_cache_change")}),
        phases=frozenset({"kv_cache_contract"}),
    ),
    ComponentRegistration(
        name="model_path",
        planner_factory=_model_path_planners,
        output_factory=_model_path_output,
        work=frozenset({("ModelPath", "model_path")}),
        phases=frozenset({"synthetic", "hf_checkpoint"}),
    ),
)


def planners(
    config_directory: Path,
    repository: Path,
    mlp_config: Path | None,
    model_config: Path,
    scenario_config: Path,
) -> tuple[Any, ...]:
    services = ComponentServices(
        config_directory,
        repository,
        mlp_config,
        model_config,
        scenario_config,
    )
    return tuple(
        planner
        for registration in REGISTRATIONS
        for planner in registration.planner_factory(services)
    )


def outputs() -> tuple[Any, ...]:
    return tuple(registration.output_factory() for registration in REGISTRATIONS)


def supported_work() -> frozenset[tuple[str, str]]:
    return frozenset(
        value for registration in REGISTRATIONS for value in registration.work
    )


def supported_phases() -> frozenset[str]:
    return frozenset(
        phase for registration in REGISTRATIONS for phase in registration.phases
    )
