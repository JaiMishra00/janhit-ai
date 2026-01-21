from qdrant_client import QdrantClient
from collections import Counter
import itertools

def inspect_collection(
    client: QdrantClient,
    collection_name: str,
    sample_points: int = 50
):
    print(f"\n=== Inspecting collection: {collection_name} ===\n")

    # ------------------------------------------------------------------
    # 1. Collection-level metadata
    # ------------------------------------------------------------------
    info = client.get_collection(collection_name)
    cfg = info.config

    print(">> Collection Config")
    print(f"- Vectors: {cfg.params.vectors}")
    print(f"- Distance: {cfg.params.vectors.distance}")
    print(f"- Shards: {cfg.params.shard_number}")
    print(f"- Replication: {cfg.params.replication_factor}")
    print(f"- On Disk Payload: {cfg.params.on_disk_payload}")
    print()

    # ------------------------------------------------------------------
    # 2. Point count
    # ------------------------------------------------------------------
    count = client.count(collection_name, exact=True).count
    print(">> Size")
    print(f"- Total points: {count}")
    print()

    # ------------------------------------------------------------------
    # 3. Scroll sample points
    # ------------------------------------------------------------------
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=sample_points,
        with_vectors=False,
        with_payload=True
    )

    if not points:
        print("No points found.")
        return

    # ------------------------------------------------------------------
    # 4. Payload key analysis
    # ------------------------------------------------------------------
    payload_keys = list(
        itertools.chain.from_iterable(
            p.payload.keys() for p in points if p.payload
        )
    )

    key_freq = Counter(payload_keys)

    print(">> Payload Keys (coverage in sample)")
    for k, v in key_freq.most_common():
        print(f"- {k}: {v}/{len(points)}")

    print()

    # ------------------------------------------------------------------
    # 5. Payload type inference
    # ------------------------------------------------------------------
    print(">> Payload Types (inferred)")
    payload_types = {}

    for p in points:
        for k, v in (p.payload or {}).items():
            payload_types.setdefault(k, set()).add(type(v).__name__)

    for k, types in payload_types.items():
        print(f"- {k}: {', '.join(types)}")

    print()

    # ------------------------------------------------------------------
    # 6. Example payload values (first occurrence)
    # ------------------------------------------------------------------
    print(">> Example Payload Values")
    shown = set()
    for p in points:
        for k, v in (p.payload or {}).items():
            if k not in shown:
                print(f"- {k}: {v}")
                shown.add(k)
        if len(shown) == len(payload_types):
            break

    print()

    # ------------------------------------------------------------------
    # 7. Safety / RAG sanity hints
    # ------------------------------------------------------------------
    print(">> Sanity Checks")
    required_keys = {"source", "created_at", "consent", "domain"}
    missing = required_keys - payload_types.keys()

    if missing:
        print(f"- Missing recommended payload fields: {missing}")
    else:
        print("- All recommended payload fields present")

    print("\n=== Inspection complete ===\n")


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    client = QdrantClient(
        url="http://localhost:6333"  # or host/port/api_key
    )

    inspect_collection(
        client,
        collection_name="md_bge_m3_dense",
        sample_points=100
    )
