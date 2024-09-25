from __future__ import annotations

import sys
from typing import Any, Iterator, TypeAlias, NamedTuple
from functools import cached_property
from contextlib import contextmanager

from collections.abc import Mapping


GlobalsNamespace: TypeAlias = dict[str, Any]
MappingNamespace: TypeAlias = Mapping[str, Any]


class NamespacesTuple(NamedTuple):
    globals: GlobalsNamespace
    """The global namespace.

    In most cases, this is a reference to the module namespace.
    """

    locals: MappingNamespace
    """
    The local namespace.

    This can be any mapping.
    """


def get_module_ns_of(obj: Any) -> dict[str, Any]:
    """Get the namespace of the module where the object is defined.

    Caution: this function does not return a copy of the module namespace, so it should not be mutated.
    The burden of enforcing this is on the caller.
    """
    module_name = getattr(obj, '__module__', None)
    if module_name:
        try:
            return sys.modules[module_name].__dict__
        except KeyError:
            # happens occasionally, see https://github.com/pydantic/pydantic/issues/2363
            return {}
    return {}


class LazyLocalNamespace(Mapping[str, Any]):
    def __init__(self, *namespaces: MappingNamespace) -> None:
        self._namespaces = namespaces

    @cached_property
    def data(self) -> Mapping[str, Any]:
        return {k: v for ns in self._namespaces for k, v in ns.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)


def ns_from(obj: Any, parent_namespace: MappingNamespace | None = None) -> NamespacesTuple:
    if isinstance(obj, type):
        locals_list: list[MappingNamespace] = []

        if parent_namespace is not None:
            locals_list.append(parent_namespace)

        locals_list.extend(
            [
                vars(obj),
                {obj.__name__: obj},
            ]
        )

        return NamespacesTuple(get_module_ns_of(obj), LazyLocalNamespace(*locals_list))

    else:
        # TBD
        return NamespacesTuple({}, LazyLocalNamespace())


class NsResolver:
    def __init__(
        self,
        namespaces_tuple: NamespacesTuple | None = None,
        fallback_namespace: MappingNamespace | None = None,
        override_namespace: MappingNamespace | None = None,
    ) -> None:
        self._base_ns_tuple = namespaces_tuple or NamespacesTuple({}, {})
        self._fallback_ns = fallback_namespace
        self._override_ns = override_namespace
        self._types_stack: list[type[Any]] = []

    @cached_property
    def eval_namespaces(self) -> NamespacesTuple:
        if not self._types_stack:
            return self._base_ns_tuple

        typ = self._types_stack[-1]

        globals_ns = get_module_ns_of(typ)
        if self._fallback_ns is not None:  # TODO check len(self._types_stack) == 1?
            globals_ns = {**self._fallback_ns, **globals_ns}

        locals_list: list[MappingNamespace] = [
            vars(typ),
            {typ.__name__: typ}
        ]
        if self._override_ns is not None:
            locals_list.append(self._override_ns)

        return NamespacesTuple(globals_ns, LazyLocalNamespace(*locals_list))

    @contextmanager
    def push(self, typ: type[Any]):
        self._types_stack.append(typ)
        del self.eval_namespaces
        try:
            yield
        finally:
            self._types_stack.pop()
            del self.eval_namespaces
