from __future__ import annotations

from collections import deque


def toposort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    indegree = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for parent, child in edges:
        outgoing[parent].append(child)
        indegree[child] += 1
    queue = deque([node for node in nodes if indegree[node] == 0])
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in outgoing[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order


def is_dag(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    return len(toposort(nodes, edges)) == len(nodes)


def find_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> list[str] | None:
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for parent, child in edges:
        adjacency[parent].append(child)

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        visiting.add(node)
        stack.append(node)
        for child in adjacency[node]:
            if child in visiting:
                if child in stack:
                    idx = stack.index(child)
                    return stack[idx:] + [child]
                return [child, node, child]
            if child not in visited:
                cycle = dfs(child)
                if cycle:
                    return cycle
        visiting.remove(node)
        visited.add(node)
        stack.pop()
        return None

    for node in nodes:
        if node in visited:
            continue
        cycle = dfs(node)
        if cycle:
            return cycle
    return None
