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
