import copy
import typing
from functools import partial
from itertools import chain, count, groupby
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import click
from pip._internal.cache import WheelCache
from pip._internal.req import InstallRequirement
from pip._internal.req.constructors import install_req_from_line
from pip._internal.req.req_tracker import (
    get_requirement_tracker,
    update_env_context_manager,
)
from pip._internal.resolution.base import BaseResolver as PipBaseResolver
from pip._internal.resolution.resolvelib.base import Candidate
from pip._internal.utils.logging import indent_log
from pip._internal.utils.temp_dir import TempDirectory, global_tempdir_manager
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name

from piptools.cache import DependencyCache
from piptools.repositories.base import BaseRepository

from .exceptions import PipToolsError
from .logging import log
from .utils import (
    UNSAFE_PACKAGES,
    format_requirement,
    format_specifier,
    is_pinned_requirement,
    is_url_requirement,
    key_from_ireq,
    remove_value,
)

green = partial(click.style, fg="green")
magenta = partial(click.style, fg="magenta")


class RequirementSummary:
    """
    Summary of a requirement's properties for comparison purposes.
    """

    def __init__(self, ireq: InstallRequirement) -> None:
        self.req = ireq.req
        self.key = key_from_ireq(ireq)
        self.extras = frozenset(ireq.extras)
        self.specifier = ireq.specifier

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return (
            self.key == other.key
            and self.specifier == other.specifier
            and self.extras == other.extras
        )

    def __hash__(self) -> int:
        return hash((self.key, self.specifier, self.extras))

    def __str__(self) -> str:
        return repr((self.key, str(self.specifier), sorted(self.extras)))


def combine_install_requirements(
    repository: BaseRepository, ireqs: Iterable[InstallRequirement]
) -> InstallRequirement:
    """
    Return a single install requirement that reflects a combination of
    all the inputs.
    """
    # We will store the source ireqs in a _source_ireqs attribute;
    # if any of the inputs have this, then use those sources directly.
    source_ireqs = []
    for ireq in ireqs:
        source_ireqs.extend(getattr(ireq, "_source_ireqs", [ireq]))

    # Optimization. Don't bother with combination logic.
    if len(source_ireqs) == 1:
        return source_ireqs[0]

    # deepcopy the accumulator so as to not modify the inputs
    combined_ireq = copy.deepcopy(source_ireqs[0])

    for ireq in source_ireqs[1:]:
        # NOTE we may be losing some info on dropped reqs here
        if combined_ireq.req is not None and ireq.req is not None:
            combined_ireq.req.specifier &= ireq.req.specifier
        combined_ireq.constraint &= ireq.constraint
        combined_ireq.extras = {*combined_ireq.extras, *ireq.extras}
        if combined_ireq.req is not None:
            combined_ireq.req.extras = set(combined_ireq.extras)

    # InstallRequirements objects are assumed to come from only one source, and
    # so they support only a single comes_from entry. This function breaks this
    # model. As a workaround, we deterministically choose a single source for
    # the comes_from entry, and add an extra _source_ireqs attribute to keep
    # track of multiple sources for use within pip-tools.
    if len(source_ireqs) > 1:
        if any(ireq.comes_from is None for ireq in source_ireqs):
            # None indicates package was directly specified.
            combined_ireq.comes_from = None
        else:
            # Populate the comes_from field from one of the sources.
            # Requirement input order is not stable, so we need to sort:
            # We choose the shortest entry in order to keep the printed
            # representation as concise as possible.
            combined_ireq.comes_from = min(
                (ireq.comes_from for ireq in source_ireqs),
                key=lambda x: (len(str(x)), str(x)),
            )
        combined_ireq._source_ireqs = source_ireqs
    return combined_ireq


class BaseResolver:
    repository: BaseRepository

    def resolve(self, max_rounds: int) -> Set[InstallRequirement]:
        raise NotImplementedError("Override in subclass")

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


class LegacyResolver(BaseResolver):
    def __init__(
        self,
        constraints: Iterable[InstallRequirement],
        existing_constraints: typing.Dict[str, InstallRequirement],
        repository: BaseRepository,
        cache: DependencyCache,
        prereleases: Optional[bool] = False,
        clear_caches: bool = False,
        allow_unsafe: bool = False,
    ) -> None:
        """
        This class resolves a given set of constraints (a collection of
        InstallRequirement objects) by consulting the given Repository and the
        DependencyCache.
        """
        self.our_constraints = set(constraints)
        self.their_constraints: Set[InstallRequirement] = set()
        self.repository = repository
        self.dependency_cache = cache
        self.prereleases = prereleases
        self.clear_caches = clear_caches
        self.allow_unsafe = allow_unsafe
        self.unsafe_constraints: Set[InstallRequirement] = set()

        options = self.repository.options
        if "legacy-resolver" not in options.deprecated_features_enabled:
            raise PipToolsError("Legacy resolver deprecated feature must be enabled.")

        # Make sure there is no enabled 2020-resolver
        options.features_enabled = remove_value(
            options.features_enabled, "2020-resolver"
        )

    @property
    def constraints(self) -> Set[InstallRequirement]:
        return set(
            self._group_constraints(chain(self.our_constraints, self.their_constraints))
        )

    def resolve(self, max_rounds: int = 10) -> Set[InstallRequirement]:
        """
        Finds concrete package versions for all the given InstallRequirements
        and their recursive dependencies.  The end result is a flat list of
        (name, version) tuples.  (Or an editable package.)

        Resolves constraints one round at a time, until they don't change
        anymore.  Protects against infinite loops by breaking out after a max
        number rounds.
        """
        if self.clear_caches:
            self.dependency_cache.clear()
            self.repository.clear_caches()

        # Ignore existing packages
        with update_env_context_manager(PIP_EXISTS_ACTION="i"):
            for current_round in count(start=1):  # pragma: no branch
                if current_round > max_rounds:
                    raise RuntimeError(
                        "No stable configuration of concrete packages "
                        "could be found for the given constraints after "
                        "{max_rounds} rounds of resolving.\n"
                        "This is likely a bug.".format(max_rounds=max_rounds)
                    )

                log.debug("")
                log.debug(magenta(f"{f'ROUND {current_round}':^60}"))
                has_changed, best_matches = self._resolve_one_round()
                log.debug("-" * 60)
                log.debug(
                    "Result of round {}: {}".format(
                        current_round,
                        "not stable" if has_changed else "stable, done",
                    )
                )
                if not has_changed:
                    break

        # Only include hard requirements and not pip constraints
        results = {req for req in best_matches if not req.constraint}

        # Filter out unsafe requirements.
        self.unsafe_constraints = set()
        if not self.allow_unsafe:
            # reverse_dependencies is used to filter out packages that are only
            # required by unsafe packages. This logic is incomplete, as it would
            # fail to filter sub-sub-dependencies of unsafe packages. None of the
            # UNSAFE_PACKAGES currently have any dependencies at all (which makes
            # sense for installation tools) so this seems sufficient.
            reverse_dependencies = self.reverse_dependencies(results)
            for req in results.copy():
                required_by = reverse_dependencies.get(req.name.lower(), set())
                if req.name in UNSAFE_PACKAGES or (
                    required_by and all(name in UNSAFE_PACKAGES for name in required_by)
                ):
                    self.unsafe_constraints.add(req)
                    results.remove(req)

        return results

    def _get_ireq_with_name(
        self,
        ireq: InstallRequirement,
        proxy_cache: Dict[InstallRequirement, InstallRequirement],
    ) -> InstallRequirement:
        """
        Return the given ireq, if it has a name, or a proxy for the given ireq
        which has been prepared and therefore has a name.

        Preparing the ireq is side-effect-ful and can only be done once for an
        instance, so we use a proxy instead. combine_install_requirements may
        use the given ireq as a template for its aggregate result, mutating it
        further by combining extras, etc. In that situation, we don't want that
        aggregate ireq to be prepared prior to mutation, since its dependencies
        will be frozen with those of only a subset of extras.

        i.e. We both want its name early (via preparation), but we also need to
        prepare it after any mutation for combination purposes. So we use a
        proxy here for the early preparation.
        """
        if ireq.name is not None:
            return ireq

        if ireq in proxy_cache:
            return proxy_cache[ireq]

        # get_dependencies has the side-effect of assigning name to ireq
        # (so we can group by the name in _group_constraints below).
        name_proxy = copy.deepcopy(ireq)
        self.repository.get_dependencies(name_proxy)
        proxy_cache[ireq] = name_proxy
        return name_proxy

    def _group_constraints(
        self, constraints: Iterable[InstallRequirement]
    ) -> Iterator[InstallRequirement]:
        """
        Groups constraints (remember, InstallRequirements!) by their key name,
        and combining their SpecifierSets into a single InstallRequirement per
        package.  For example, given the following constraints:

            Django<1.9,>=1.4.2
            django~=1.5
            Flask~=0.7

        This will be combined into a single entry per package:

            django~=1.5,<1.9,>=1.4.2
            flask~=0.7

        """
        constraints = list(constraints)

        cache: Dict[InstallRequirement, InstallRequirement] = {}

        def key_from_ireq_with_name(ireq: InstallRequirement) -> str:
            """
            See _get_ireq_with_name for context.

            We use a cache per call here because it should only be necessary
            the first time an ireq is passed here (later on in the round, it
            will be prepared and dependencies for it calculated), but we can
            save time by reusing the proxy between the sort and groupby calls
            below.
            """
            return key_from_ireq(self._get_ireq_with_name(ireq, cache))

        # Sort first by name, i.e. the groupby key. Then within each group,
        # sort editables first.
        # This way, we don't bother with combining editables, since the first
        # ireq will be editable, if one exists.
        for _, ireqs in groupby(
            sorted(
                constraints,
                key=(lambda x: (key_from_ireq_with_name(x), not x.editable)),
            ),
            key=key_from_ireq_with_name,
        ):
            yield combine_install_requirements(self.repository, ireqs)

    def _resolve_one_round(self) -> Tuple[bool, Set[InstallRequirement]]:
        """
        Resolves one level of the current constraints, by finding the best
        match for each package in the repository and adding all requirements
        for those best package versions.  Some of these constraints may be new
        or updated.

        Returns whether new constraints appeared in this round.  If no
        constraints were added or changed, this indicates a stable
        configuration.
        """
        # Sort this list for readability of terminal output
        constraints = sorted(self.constraints, key=key_from_ireq)

        log.debug("Current constraints:")
        with log.indentation():
            for constraint in constraints:
                log.debug(str(constraint))

        log.debug("")
        log.debug("Finding the best candidates:")
        with log.indentation():
            best_matches = {self.get_best_match(ireq) for ireq in constraints}

        # Find the new set of secondary dependencies
        log.debug("")
        log.debug("Finding secondary dependencies:")

        their_constraints: List[InstallRequirement] = []
        with log.indentation():
            for best_match in best_matches:
                their_constraints.extend(self._iter_dependencies(best_match))
        # Grouping constraints to make clean diff between rounds
        theirs = set(self._group_constraints(their_constraints))

        # NOTE: We need to compare RequirementSummary objects, since
        # InstallRequirement does not define equality
        diff = {RequirementSummary(t) for t in theirs} - {
            RequirementSummary(t) for t in self.their_constraints
        }
        removed = {RequirementSummary(t) for t in self.their_constraints} - {
            RequirementSummary(t) for t in theirs
        }

        has_changed = len(diff) > 0 or len(removed) > 0
        if has_changed:
            log.debug("")
            log.debug("New dependencies found in this round:")
            with log.indentation():
                for new_dependency in sorted(diff, key=key_from_ireq):
                    log.debug(f"adding {new_dependency}")
            log.debug("Removed dependencies in this round:")
            with log.indentation():
                for removed_dependency in sorted(removed, key=key_from_ireq):
                    log.debug(f"removing {removed_dependency}")

        # Store the last round's results in the their_constraints
        self.their_constraints = theirs
        return has_changed, best_matches

    def get_best_match(self, ireq: InstallRequirement) -> InstallRequirement:
        """
        Returns a (pinned or editable) InstallRequirement, indicating the best
        match to use for the given InstallRequirement (in the form of an
        InstallRequirement).

        Example:
        Given the constraint Flask>=0.10, may return Flask==0.10.1 at
        a certain moment in time.

        Pinned requirements will always return themselves, i.e.

            Flask==0.10.1 => Flask==0.10.1

        """
        if ireq.editable or is_url_requirement(ireq):
            # NOTE: it's much quicker to immediately return instead of
            # hitting the index server
            best_match = ireq
        elif is_pinned_requirement(ireq):
            # NOTE: it's much quicker to immediately return instead of
            # hitting the index server
            best_match = ireq
        elif ireq.constraint:
            # NOTE: This is not a requirement (yet) and does not need
            # to be resolved
            best_match = ireq
        else:
            best_match = self.repository.find_best_match(
                ireq, prereleases=self.prereleases
            )

        # Format the best match
        log.debug(
            "found candidate {} (constraint was {})".format(
                format_requirement(best_match), format_specifier(ireq)
            )
        )
        best_match.comes_from = ireq.comes_from
        if hasattr(ireq, "_source_ireqs"):
            best_match._source_ireqs = ireq._source_ireqs
        return best_match

    def _iter_dependencies(
        self, ireq: InstallRequirement
    ) -> Iterator[InstallRequirement]:
        """
        Given a pinned, url, or editable InstallRequirement, collects all the
        secondary dependencies for them, either by looking them up in a local
        cache, or by reaching out to the repository.

        Editable requirements will never be looked up, as they may have
        changed at any time.
        """
        # Pip does not resolve dependencies of constraints. We skip handling
        # constraints here as well to prevent the cache from being polluted.
        # Constraints that are later determined to be dependencies will be
        # marked as non-constraints in later rounds by
        # `combine_install_requirements`, and will be properly resolved.
        # See https://github.com/pypa/pip/
        # blob/6896dfcd831330c13e076a74624d95fa55ff53f4/src/pip/_internal/
        # legacy_resolve.py#L325
        if ireq.constraint:
            return

        if ireq.editable or is_url_requirement(ireq):
            yield from self.repository.get_dependencies(ireq)
            return
        elif not is_pinned_requirement(ireq):
            raise TypeError(f"Expected pinned or editable requirement, got {ireq}")

        # Now, either get the dependencies from the dependency cache (for
        # speed), or reach out to the external repository to
        # download and inspect the package version and get dependencies
        # from there
        if ireq not in self.dependency_cache:
            log.debug(
                f"{format_requirement(ireq)} not in cache, need to check index",
                fg="yellow",
            )
            dependencies = self.repository.get_dependencies(ireq)
            self.dependency_cache[ireq] = sorted(str(ireq.req) for ireq in dependencies)

        # Example: ['Werkzeug>=0.9', 'Jinja2>=2.4']
        dependency_strings = self.dependency_cache[ireq]
        log.debug(
            "{:25} requires {}".format(
                format_requirement(ireq),
                ", ".join(sorted(dependency_strings, key=lambda s: s.lower())) or "-",
            )
        )
        for dependency_string in dependency_strings:
            yield install_req_from_line(
                dependency_string, constraint=ireq.constraint, comes_from=ireq
            )

    def reverse_dependencies(
        self, ireqs: Iterable[InstallRequirement]
    ) -> Dict[str, Set[str]]:
        non_editable = [
            ireq for ireq in ireqs if not (ireq.editable or is_url_requirement(ireq))
        ]
        return self.dependency_cache.reverse_dependencies(non_editable)


class Resolver(BaseResolver):
    """
    Uses pip's 2020 resolver.
    """

    def __init__(
        self,
        constraints: Iterable[InstallRequirement],
        existing_constraints: typing.Dict[str, InstallRequirement],
        repository: BaseRepository,
        allow_unsafe: bool = False,
        **kwargs: Any,
    ) -> None:
        self.constraints = list(constraints)
        self.repository = repository
        self.allow_unsafe = allow_unsafe

        options = self.options = self.repository.options
        self.session = self.repository.session
        self.finder = self.repository.finder
        self.command = self.repository.command
        self.unsafe_constraints: Set[InstallRequirement] = set()

        self.existing_constraints = existing_constraints
        self._constraints_map = {key_from_ireq(ireq): ireq for ireq in constraints}

        if "2020-resolver" not in options.features_enabled:
            raise PipToolsError("2020 resolver feature must be enabled.")

        # Make sure there is no enabled legacy resolver
        options.deprecated_features_enabled = remove_value(
            options.deprecated_features_enabled, "legacy-resolver"
        )

    def resolve(self, max_rounds: int = 10) -> Set[InstallRequirement]:
        with get_requirement_tracker() as req_tracker, global_tempdir_manager(), indent_log(), update_env_context_manager(  # noqa: E501
            PIP_EXISTS_ACTION="i"
        ):
            # Mark direct/primary/user_supplied packages
            for ireq in self.constraints:
                ireq.user_supplied = True

            # Pass compiled requirements from `requirements.txt` as constraints to resolver
            existing_constraints = list(self.existing_constraints.values())
            for ireq in existing_constraints:
                ireq.constraint = True
                ireq.user_supplied = False
                ireq._is_existing_pin = True

            self.constraints.extend(existing_constraints)

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
        self, resolver: PipBaseResolver
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
        self, resolver: PipBaseResolver, candidate: Candidate
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

        # Determine a pin operator
        version_pin_operator = "=="
        version_as_str = str(candidate.version)
        for specifier in ireq.specifier:
            if specifier.operator == "===" and specifier.version == version_as_str:
                version_pin_operator = "==="
                break

        # Prepare pinned install requirement. Copy from current the install requirement,
        # so that it could be mutated later.
        pinned_ireq = copy.deepcopy(ireq)

        # Canonicalize name and pin requirement to a resolved version
        pinned_ireq.req.name = canonicalize_name(ireq.name)
        pinned_ireq.req.specifier = SpecifierSet(
            f"{version_pin_operator}{candidate.version}"
        )

        # Prepare install requirement parents for annotation
        pinned_ireq._required_by = tuple(
            parent_name
            for parent_name in resolver._result.graph.iter_parents(candidate.name)
            if parent_name is not None
        )

        # Prepare install requirement sources for annotation
        ireq_key = key_from_ireq(pinned_ireq)
        source_ireq = self._constraints_map.get(ireq_key)
        if source_ireq is not None and ireq_key not in self.existing_constraints:
            pinned_ireq._source_ireqs = [source_ireq]

        return pinned_ireq
