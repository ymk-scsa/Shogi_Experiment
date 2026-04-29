from __future__ import annotations

import heapq
from itertools import count
from typing import Optional

import cshogi


class NPLSNode:
    """NPLS 探索の1ノード（ルートからの1手・局面・評価・優先度を保持）。"""

    def __init__(
        self,
        depth: int = 1,
        moves: Optional[list[int]] = None,
        value: float = 0.0,
        total_value: float = 0.0,
        value_variance: float = 0.0,
        policy: float = 0.0,
        total_policy: float = 1.0,
        board: Optional[cshogi.Board] = None,
        priority: float = 0.0,
    ) -> None:
        self.depth: int = depth
        self.moves: list[int] = moves if moves is not None else []
        self.value: float = value
        self.total_value: float = total_value
        self.value_variance: float = value_variance
        self.policy: float = policy
        self.total_policy: float = total_policy
        self.board: cshogi.Board = board if board is not None else cshogi.Board()
        self.priority: float = priority

    def compute_priority(self) -> float:
        return (3.0 / self.depth) + (self.policy * 2.0 + self.total_policy) + (self.total_value / self.depth) + (self.value_variance / self.depth)


class NPLSNodeTree:
    """優先度が最大のノードを取り出すオープンリスト（最小ヒープで負の優先度を保持）。"""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, NPLSNode]] = []
        self._seq = count()

    def clear(self) -> None:
        self._heap.clear()

    def push(self, node: NPLSNode) -> None:
        heapq.heappush(self._heap, (-node.priority, next(self._seq), node))

    def pop_max(self, count: int = 1024) -> Optional[list[NPLSNode]]:
        return [node for _, _, node in heapq.nlargest(min(count, len(self._heap)), self._heap)]

    def __len__(self) -> int:
        return len(self._heap)
    
    def recycle_nodes(self, root_first_move: int, value: float, policy: float) -> None:
        """ヒープ要素は (-priority, seq, NPLSNode)。先頭手が一致するノードだけ PV を1手すく。

        一致しない場合はヒープをそのまま保持する（呼び出し側が手数など誤った値を渡しても破壊しない）。
        """
        if not self._heap:
            return
        new_heap: list[tuple[float, int, NPLSNode]] = []
        for neg_pri, seq, n in self._heap:
            if n.moves and n.moves[0] == root_first_move:
                rest = n.moves[1:]
                if not rest:
                    continue
                new_heap.append(
                    (
                        neg_pri,
                        seq,
                        NPLSNode(
                            depth=max(1, n.depth - 1),
                            moves=rest,
                            value=value,
                            total_value=n.total_value,
                            policy=policy,
                            total_policy=n.total_policy,
                            board=n.board,
                            priority=n.priority,
                        ),
                    )
                )
            else:
                new_heap.append((neg_pri, seq, n))
        self._heap = new_heap
