"""Consume append-only local market snapshots from a feed publisher."""

from .paper import PaperEngine


class ForwardBook:
    def __init__(self, dataset, policy, config, features):
        self.engine = PaperEngine(dataset.meta, policy, config)
        self.bars = []
        self.documents = {}
        self.features = []
        self.identity = self.identity_of(dataset)
        self.advance(dataset, features)

    @staticmethod
    def identity_of(dataset):
        return tuple(dataset.meta[k] for k in ("instrument", "currency", "venue", "lot_size",
                                               "price_basis", "calendar_version"))

    def advance(self, dataset, features):
        if self.identity_of(dataset) != self.identity or dataset.bars[:len(self.bars)] != self.bars:
            raise ValueError("listing, valuation rules or processed bars changed")
        docs = {doc["id"]: doc for doc in dataset.documents}
        if any(docs.get(key) != old for key, old in self.documents.items()):
            raise ValueError("existing source document changed; append a new revision")
        if features[:len(self.features)] != self.features:
            raise ValueError("existing model features changed")
        if self.bars:
            cutoff = self.bars[-1].available_at
            if (any(doc["available_at"] <= cutoff for key, doc in docs.items() if key not in self.documents)
                    or any(f.observed_at <= cutoff for f in features[len(self.features):])):
                raise ValueError("new evidence backdated across a processed decision")
        for bar in dataset.bars[len(self.bars):]:
            self.engine.observe(bar, dataset.documents, features)
        changed = len(dataset.bars) != len(self.bars)
        self.bars, self.documents, self.features = dataset.bars.copy(), docs, list(features)
        return changed
