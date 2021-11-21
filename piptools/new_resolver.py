from typing import Any, Dict, Iterable, Optional, Set

from pip._internal.cache import WheelCache
from pip._internal.req import InstallRequirement
from pip._internal.req.req_tracker import (
    get_requirement_tracker,
    update_env_context_manager,
)
from pip._internal.resolution.base import BaseResolver
from pip._internal.resolution.resolvelib.base import Candidate
from pip._internal.utils.logging import indent_log
from pip._internal.utils.temp_dir import TempDirectory, global_tempdir_manager
from pip._vendor.packaging.specifiers import SpecifierSet

from piptools.logging import log
from piptools.repositories.base import BaseRepository
from piptools.utils import UNSAFE_PACKAGES, key_from_ireq


class NewResolver:
    # FIXME: needs appropriate module and name
    def __init__(
        self,
        constraints: Iterable[InstallRequirement],
        repository: BaseRepository,
        allow_unsafe: bool = False,
        **kwargs: Any,
    ) -> None:
        self.constraints = constraints
        self.repository = repository
        self.allow_unsafe = allow_unsafe

        self.options = self.repository.options
        self.session = self.repository.session
        self.finder = self.repository.finder
        self.command = self.repository.command
        self.unsafe_constraints: Set[InstallRequirement] = set()

        self._constraints_map = {key_from_ireq(ireq): ireq for ireq in constraints}

    def resolve(self, max_rounds: int = 10) -> Set[InstallRequirement]:
        with get_requirement_tracker() as req_tracker, global_tempdir_manager(), indent_log(), update_env_context_manager(  # noqa: E501
            PIP_EXISTS_ACTION="i"
        ):
            wheel_cache = WheelCache(
                self.options.cache_dir, self.options.format_control
            )

            temp_dir = TempDirectory(
                delete=not self.options.no_clean,
                kind="resolve",
                globally_managed=True,
            )

            # If any requirement has hash options, enable hash checking.
            if any(req.has_hash_options for req in self.constraints):
                self.options.require_hashes = True

            preparer = self.command.make_requirement_preparer(
                temp_build_dir=temp_dir,
                options=self.options,
                req_tracker=req_tracker,
                session=self.session,
                finder=self.finder,
                use_user_site=False,
            )

            resolver = self.command.make_resolver(
                preparer=preparer,
                finder=self.finder,
                options=self.options,
                wheel_cache=wheel_cache,
                use_user_site=False,
                ignore_installed=True,
                ignore_requires_python=False,
                force_reinstall=False,
                use_pep517=self.options.use_pep517,
                upgrade_strategy="to-satisfy-only",
            )

            self.command.trace_basic_info(self.finder)

            resolver.resolve(
                root_reqs=self.constraints,
                check_supported_wheels=not self.options.target_dir,
            )

        return self._get_install_requirements(resolver)

    def _get_install_requirements(
        self, resolver: BaseResolver
    ) -> Set[InstallRequirement]:
        """Returns a set of install requirements from resolver results."""
        reqs = set()
        for candidate in resolver._result.mapping.values():
            ireq = self._get_install_requirement_from_candidate(
                resolver=resolver,
                candidate=candidate,
            )
            if ireq is None:
                continue
            reqs.add(ireq)
        return reqs

    def _get_install_requirement_from_candidate(
        self, resolver: BaseResolver, candidate: Candidate
    ) -> Optional[InstallRequirement]:
        ireq = candidate.get_install_requirement()
        if ireq is None:
            return None

        # Filter out unsafe requirements. This logic is incomplete, as it would
        # fail to filter sub-dependencies of unsafe packages. None of the
        # UNSAFE_PACKAGES currently have any dependencies at all (which makes sense
        # for installation tools) so this seems sufficient.
        if not self.allow_unsafe and ireq.name in UNSAFE_PACKAGES:
            self.unsafe_constraints.add(ireq)
            return None

        # Detect pin operator
        version_pin_operator = "=="
        version_as_str = str(candidate.version)
        for specifier in ireq.specifier:
            if specifier.operator == "===" and specifier.version == version_as_str:
                version_pin_operator = "==="
                break

        # Override version specifier
        ireq.req.specifier = SpecifierSet(f"{version_pin_operator}{candidate.version}")

        # Prepare install requirement parents for annotation
        ireq._required_by = tuple(
            parent_name
            for parent_name in resolver._result.graph.iter_parents(candidate.name)
            if parent_name is not None
        )

        # Prepare install requirement sources for annotation
        source_ireq = self._constraints_map.get(key_from_ireq(ireq))
        if source_ireq is not None and (
            not hasattr(source_ireq, "_is_existing_pin")
            or not source_ireq._is_existing_pin
        ):
            ireq._source_ireqs = [source_ireq]

        return ireq

    def resolve_hashes(
        self, ireqs: Set[InstallRequirement]
    ) -> Dict[InstallRequirement, Set[str]]:
        """
        Finds acceptable hashes for all of the given InstallRequirements.
        """
        log.debug("")
        log.debug("Generating hashes:")
        with self.repository.allow_all_wheels(), log.indentation():
            return {ireq: self.repository.get_hashes(ireq) for ireq in ireqs}
