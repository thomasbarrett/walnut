"""Prefix sharing: what two prompts that begin alike can avoid paying twice.

A page of key/value cache is a pure function of the tokens before it, so two
prompts that agree on their first n tokens agree on the cache those tokens
produce. A trie over 256-token blocks — one node per page, which is the
granularity paging already forced — is enough to find that agreement and hand
the second prompt the pages the first one built.

On a hybrid model that is not enough, and the shortfall is the whole design.
Qwen3.5 is eighteen linear-attention layers to six full-attention ones, and a
linear layer's state is a `walnut.cache.state.StateCache`: one running summary
per sequence, with no cell to point a second sequence at. Reusing pages without
it saves only the six layers' *writes*, because the other eighteen still have
to be walked from the beginning to arrive at the right state — and walking them
means running the whole stack, attention included. Shared pages alone are worth
approximately nothing here.

So what this caches is the state, and the pages ride along. A `Checkpoint` is
every recurrent layer's row copied out at a page boundary; restoring one into a
fresh row, alongside that prefix's pages, puts a new sequence exactly where the
old one stood, and its prefill starts from there.

The costs are not close to each other, which sets the policy. A page is 3 MiB
across the six attention layers; a checkpoint is 18.6 MiB across the eighteen
recurrent ones — six pages, or about 1550 tokens of cache, for one boundary.
Checkpointing every node would spend six times more on the state than on the
cache it exists to skip. So pages are kept for every prompt, because they are
cheap and the trie needs them anyway, and a checkpoint is taken only at a node
a *second* prompt has reached: proof that the prefix is shared before paying
for it. `Checkpoints` holds a fixed number of them and evicts by least-recently
used.

What that yields is a prefix warm from the third request rather than the
second. The first inserts the nodes; the second matches them, which is what
marks the boundary worth keeping, and takes the checkpoint as its own prefill
passes through; the third and everything after start there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from walnut.cache.pool import PAGE_SIZE, CachePool

#: A page's worth of token ids, which is what a trie node is keyed by.
Block = tuple[int, ...]


def blocks(tokens: list[int]) -> list[Block]:
    """``tokens`` as whole pages, dropping the partial one at the end.

    Only whole pages: a partial page is still being written by the sequence
    that holds it, so there is nothing complete to share.
    """
    whole = len(tokens) // PAGE_SIZE
    return [tuple(tokens[i * PAGE_SIZE : (i + 1) * PAGE_SIZE]) for i in range(whole)]


@dataclass
class Node:
    """One page of a shared prefix: the tokens, the page, and who wants it."""

    #: The token ids this node covers — a whole page of them, and the key its
    #: parent finds it by.
    block: Block
    #: The physical page holding those tokens' keys and values. Owned by the
    #: tree, not by any sequence.
    page: int
    parent: Node | None
    children: dict[Block, Node] = field(default_factory=dict)
    #: Sequences currently reading this page. A node with references cannot be
    #: evicted; one without can, oldest first.
    refs: int = 0
    #: Prompts that have matched down through this node. The second is what
    #: makes a checkpoint here worth its memory.
    hits: int = 0
    #: Index into `Checkpoints`, once one has been taken at this boundary.
    checkpoint: int | None = None
    #: Last use, on the tree's own clock, for eviction order.
    clock: int = 0

    @property
    def depth(self) -> int:
        """Pages from the root, so ``depth * PAGE_SIZE`` tokens end here."""
        return 0 if self.parent is None else self.parent.depth + 1


class Checkpoints:
    """A fixed number of recurrent-state snapshots, evicted by least recent use.

    Preallocated, and copied into rather than allocated per save: a checkpoint
    is every recurrent layer's row at once, and allocating one on the
    time-to-first-token path would put a large allocation exactly where it is
    least affordable.

    Full means full. A checkpoint is never dropped to make room for a new one,
    only when the node holding it leaves the trie — which is the trie's own
    least-recently-used decision, taken over a resource that runs out sooner.
    Replacing on demand looks like the obvious policy and is a trap: when more
    prefixes want a checkpoint than there are slots, every candidate evicts one
    that had not yet been used, so the store churns at full cost and returns
    nothing. Measured at 64 prefixes into 16 slots, that ran 7% *slower* than
    keeping no checkpoints at all. Refusing instead means the first prefixes to
    prove themselves keep their slots and everything else simply prefills,
    which is the behaviour of not having a cache rather than of having a bad
    one.
    """

    def __init__(self, pool: CachePool, capacity: int) -> None:
        template = pool.carried(0)
        self.capacity = 0 if not template else capacity
        self.slots = [
            [torch.empty_like(tensor) for tensor in template]
            for _ in range(self.capacity)
        ]
        self._free = list(range(self.capacity))
        #: Which node holds each taken slot, so eviction can clear its claim.
        self._held: dict[int, Node] = {}

    def __len__(self) -> int:
        return self.capacity - len(self._free)

    @property
    def bytes(self) -> int:
        """What the whole store costs, which is the figure worth reporting."""
        if not self.slots:
            return 0
        one = sum(tensor.numel() * tensor.element_size() for tensor in self.slots[0])
        return one * self.capacity

    def save(self, pool: CachePool, row: int, node: Node, clock: int) -> bool:
        """Copy ``row``'s recurrent state in, as ``node``'s checkpoint.

        False when the store is full or this node already has one, which is a
        reason to keep prefilling rather than an error.
        """
        if not self.capacity or node.checkpoint is not None or not self._free:
            return False
        index = self._free.pop()
        torch._foreach_copy_(self.slots[index], pool.carried(row))
        node.checkpoint = index
        node.clock = clock
        self._held[index] = node
        return True

    def load(self, pool: CachePool, row: int, index: int) -> None:
        """Copy checkpoint ``index`` out into ``row``'s recurrent state."""
        torch._foreach_copy_(pool.carried(row), self.slots[index])

    def release(self, index: int) -> None:
        """Give a checkpoint's slot back, once its node is gone."""
        node = self._held.pop(index, None)
        if node is not None:
            node.checkpoint = None
        self._free.append(index)

    @property
    def full(self) -> bool:
        """Whether a new prefix could still be given a checkpoint."""
        return not self._free


@dataclass(frozen=True)
class Placement:
    """Where a newly admitted sequence stands before it has run anything."""

    #: The pool row it holds.
    row: int
    #: Prompt tokens already in cache and already in this row's state, so
    #: prefill starts here rather than at zero.
    prefilled: int
    #: Position at which this sequence should check in a checkpoint on its way
    #: past, or None. The scheduler ends a prefill chunk exactly here so the
    #: state is caught at the boundary rather than somewhere after it.
    checkpoint_at: int | None


@dataclass
class Holding:
    """What one live row borrowed and what it owns, until it retires."""

    #: Every node the prompt matched, deepest last. Not all of them are being
    #: read — see `shared` — but all of them are places a checkpoint could go.
    path: list[Node]
    #: How many of `path` this row is actually reading, and holds a reference
    #: on. The rest it recomputes into pages of its own, because writing a page
    #: another sequence is reading is only safe if the two agree bit for bit,
    #: and two runs that chunked their prefill differently need not.
    shared: int
    #: Pages the pool gave this row, in order, the first covering logical page
    #: `shared`.
    owned: list[int]


class PrefixCache:
    """A trie of shared prefixes over `CachePool`, and the allocator above it.

    Everything the scheduler used to ask the pool for it asks here instead:
    `acquire` gives a sequence its row and tells it how much of its prompt it
    can skip, and `release` takes the row back and offers what the sequence
    built to the next prompt that begins the same way.

    Pages the tree holds are not free, and are not counted as free, but they
    are reclaimable: when a reservation cannot be met, the least recently used
    leaves no sequence is reading are dropped until it can. So the cache never
    costs a request its admission — only, at worst, the sharing it might have
    had.
    """

    def __init__(self, pool: CachePool, checkpoints: int = 16) -> None:
        self.pool = pool
        self.checkpoints = Checkpoints(pool, checkpoints)
        self.root = Node(block=(), page=CachePool.SCRATCH, parent=None)
        #: Monotonic, and the only ordering eviction needs — wall time would
        #: say the same thing about a queue only this ever touches.
        self.clock = 0
        self._live: dict[int, Holding] = {}
        self.nodes = 0
        self.hits = 0
        self.tokens_saved = 0
        #: Prompts offered and their total length, so a hit rate has a
        #: denominator. Without one, "16 hits" says nothing about whether the
        #: cache was working or the workload merely had 16 requests.
        self.requests = 0
        self.prompt_tokens = 0

    # -- admission ---------------------------------------------------------

    def acquire(self, prompt: list[int], wanted: int) -> Placement | None:
        """Take a row for ``prompt``, reusing whatever prefix is already here.

        ``wanted`` is prompt plus completion, the whole extent this sequence
        will need. None when the pool has no room even after eviction, which is
        the caller's signal to wait rather than to fail.

        The prefix that is *usable* is not the prefix that matches. Matching
        finds pages, and pages alone let a sequence skip nothing on a hybrid
        model; the reuse point is the deepest matched node that also holds a
        checkpoint. What matched past it is still worth knowing, because that
        is where the next checkpoint should go.

        The row check comes first because the caller is a loop. A request that
        does not fit is offered again whenever one retires, and matching the
        whole prompt into pages before discovering there is nowhere to put it
        measured 15% of throughput at batch 32. Nothing below here is worth
        doing for a request that has no row to go in.
        """
        if not self.pool.free_rows:
            return None
        self.clock += 1
        self.requests += 1
        self.prompt_tokens += len(prompt)
        path = self._match(blocks(prompt))
        for node in path:
            node.hits += 1
            node.clock = self.clock

        shared = self._reuse_point(path, len(prompt))
        row = self._reserve(wanted, shared)
        if row is None:
            return None

        for node in path[:shared]:
            node.refs += 1
        self._live[row] = Holding(path, shared, list(self.pool.held(row)))
        self._lend(row, path[:shared])

        if shared:
            index = path[shared - 1].checkpoint
            assert index is not None
            self.checkpoints.load(self.pool, row, index)
            self.hits += 1
            self.tokens_saved += shared * PAGE_SIZE
        else:
            # Nothing to stand on, so the row starts from no context at all.
            self.pool.reset(row)

        return Placement(row, shared * PAGE_SIZE, self._checkpoint_point(path, shared))

    def _match(self, wanted: list[Block]) -> list[Node]:
        """The nodes ``wanted`` walks through, longest agreeing run first."""
        node, path = self.root, []
        for block in wanted:
            child = node.children.get(block)
            if child is None:
                break
            node = child
            path.append(child)
        return path

    @staticmethod
    def _reuse_point(path: list[Node], prompt: int) -> int:
        """Pages a new sequence can start past: matched *and* checkpointed.

        Never the whole prompt. A sequence that skipped every token of it would
        have nothing left to run and no logits to sample its first token from,
        so the last page of an exactly repeated prompt is always recomputed.
        """
        for depth in range(min(len(path), (prompt - 1) // PAGE_SIZE), 0, -1):
            if path[depth - 1].checkpoint is not None:
                return depth
        return 0

    def _checkpoint_point(self, path: list[Node], shared: int) -> int | None:
        """Where this sequence should stop and check its state in, if anywhere.

        The deepest node a second prompt has now reached and that has no
        checkpoint yet — deepest, because that is the most a later prompt gets
        to skip. Past the reuse point by construction: a node with a checkpoint
        is one this sequence would have started from.
        """
        if self.checkpoints.full:
            return None
        for node in reversed(path[shared:]):
            if node.checkpoint is None and node.hits >= 2:
                return node.depth * PAGE_SIZE
        return None

    def _reserve(self, wanted: int, shared: int) -> int | None:
        """A row plus pages for everything past ``shared`` pages of it.

        Evicts what the tree is holding, oldest first, until the pool can meet
        it — so a prefix nobody is reading never outranks a request that has
        actually arrived.

        Only ever for want of *pages*, which is what the free-row guard is for.
        A full batch also fails to reserve, and every iteration of the
        scheduler's loop tries again while a request waits; without the guard
        each of those attempts evicted until the tree was empty, so a batch
        that stayed full destroyed the cache it was supposed to be filling.
        Eviction is an answer to one shortage and not to the other.
        """
        row = self.pool.reserve(wanted, shared)
        while row is None and self.pool.free_rows and self._evict():
            row = self.pool.reserve(wanted, shared)
        return row

    def _lend(self, row: int, borrowed: list[Node]) -> None:
        """Point ``row``'s block table at the pages it is borrowing."""
        if not borrowed:
            return
        table = self.pool.block_table[row]
        table[: len(borrowed)] = torch.tensor(
            [node.page for node in borrowed], dtype=torch.int32, device=table.device
        )

    # -- retirement --------------------------------------------------------

    def release(self, row: int, tokens: list[int]) -> None:
        """Give ``row`` back, offering what it wrote to the next prompt.

        ``tokens`` is the whole sequence — prompt and completion — because a
        follow-up turn resends both, and the pages holding the answer are as
        shareable as the pages holding the question.
        """
        self.clock += 1
        live = self._live.pop(row, None)
        if live is None:
            self.pool.release(row)
            return
        for node in live.path[: live.shared]:
            node.refs -= 1
            node.clock = self.clock
        self.pool.release(row, keep=self._graft(blocks(tokens), live))

    def _graft(self, wanted: list[Block], live: Holding) -> set[int]:
        """Hang this sequence's pages off the tree; return the ones it kept.

        Only the blocks past what it matched are on offer: where the tree
        already has a block it already has a page for it, and this sequence's
        copy — which it wrote itself, having declined to share — is redundant
        and goes back to the pool.
        """
        node = live.path[-1] if live.path else self.root
        kept = set()
        for depth in range(len(live.path), len(wanted)):
            index = depth - live.shared
            if index >= len(live.owned):
                break
            block = wanted[depth]
            child = node.children.get(block)
            if child is None:
                child = Node(block=block, page=live.owned[index], parent=node)
                child.hits, child.clock = 1, self.clock
                node.children[block] = child
                self.nodes += 1
                kept.add(child.page)
            node = child
        return kept

    # -- checkpoints -------------------------------------------------------

    def checkpoint(self, row: int, position: int) -> None:
        """Check ``row``'s recurrent state in at ``position``, if a node wants it.

        Called by the scheduler when a prefill chunk lands exactly on the
        boundary `Placement.checkpoint_at` named. Does nothing if the node has
        since been given a checkpoint by another sequence, or if the store is
        full of checkpoints that are all being read.
        """
        live = self._live.get(row)
        depth = position // PAGE_SIZE
        if live is None or not 0 < depth <= len(live.path):
            return
        self.checkpoints.save(self.pool, row, live.path[depth - 1], self.clock)

    # -- eviction ----------------------------------------------------------

    def _evict(self) -> bool:
        """Drop the least recently used leaf no sequence is reading through.

        A leaf, because a node with children is the prefix of a longer one that
        is still held: taking its page would leave the tree naming cache that
        is no longer there. Evicting leaves repeatedly walks a branch back from
        its tip, which is the order a prefix goes cold in anyway.
        """
        stale = [node for node in self._leaves() if node.refs == 0]
        if not stale:
            return False
        self._drop(min(stale, key=lambda node: node.clock))
        return True

    def _drop(self, node: Node) -> bool:
        """Unhook one leaf and give its page and checkpoint back."""
        if node.refs or node.children or node.parent is None:
            return False
        if node.checkpoint is not None:
            self.checkpoints.release(node.checkpoint)
        del node.parent.children[node.block]
        self.pool.free_page(node.page)
        self.nodes -= 1
        return True

    def _leaves(self) -> list[Node]:
        """Every childless node, which is every page eviction may take."""

        def walk(node: Node) -> list[Node]:
            if not node.children:
                return [node] if node is not self.root else []
            return [leaf for child in node.children.values() for leaf in walk(child)]

        return walk(self.root)

    def reset(self) -> None:
        """Forget every prefix, returning the pages to the pool.

        What a benchmark calls between a warm-up and the run it is timing. A
        warm-up that sent the same prompts leaves the trie holding exactly the
        prefixes the measurement is about to send, so without this the run
        reports a half-warm cache under either label — the trap that makes a
        prefix-cache number meaningless. Rows in flight keep what they borrowed;
        only what nothing is reading is dropped.
        """
        # Round by round, because dropping a leaf makes its parent one. Stops
        # when a pass drops nothing, which is what a trie pinned down to its
        # live rows looks like.
        dropped = True
        while dropped:
            dropped = False
            for node in self._leaves():
                dropped = self._drop(node) or dropped
        self.clear_counters()

    def clear_counters(self) -> None:
        """Zero what the cache has done, without forgetting what it holds.

        The other half of a benchmark's bookkeeping: a run measuring a *warm*
        cache wants the trie kept and the tally restarted, so that each round
        reports its own hits rather than every round before it.
        """
        self.requests = 0
        self.prompt_tokens = 0
        self.hits = 0
        self.tokens_saved = 0

    # -- reporting ---------------------------------------------------------

    @property
    def cached_tokens(self) -> int:
        """Tokens the tree is holding cache for, reclaimable but not free."""
        return self.nodes * PAGE_SIZE

    def stats(self) -> dict[str, int | float]:
        """What the cache did, in the terms a benchmark should be reading.

        ``hit_rate`` is over prompt tokens rather than over requests, because
        that is what the cache actually saves and what a request feels: half of
        one long prompt skipped is worth more than the whole of a short one.
        """
        return {
            "requests": self.requests,
            "hits": self.hits,
            "prompt_tokens": self.prompt_tokens,
            "tokens_saved": self.tokens_saved,
            "hit_rate": (
                self.tokens_saved / self.prompt_tokens if self.prompt_tokens else 0.0
            ),
            "checkpoints": len(self.checkpoints),
            "checkpoint_capacity": self.checkpoints.capacity,
            "cached_tokens": self.cached_tokens,
        }
