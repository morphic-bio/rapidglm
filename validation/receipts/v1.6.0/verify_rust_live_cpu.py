"""Independent receipt checks for the Rust-only live-process CPU contribution."""
import math

def verify(row):
    try:
        e=row["container_cpu_audit"]
        assert e["provider"]=="rust-live-descendants-v1"
        assert e["start_descendants"]==[]
        assert e["sampler_affinity"]==list(range(6))
        assert e["selected_cpus"]==row["cpus"]
        assert e["process_identity"]["pid"]==row["pid"] and e["process_identity"]["start_ticks"]>0
        assert row["accounting_start"]["rust_live_worker_start"]==dict(descendants=[],attempt_id=e["attempt_id"])
        assert row["accounting_end"]["rust_live_worker_end"]==dict(provider=e["provider"],live_cpu_seconds=e["cpu_seconds"])
        assert row["parent_cpu_counted"]
        assert row["accounting_start"]["affinity"]==row["cpus"]==row["accounting_end"]["affinity"]
        assert row["accounting_end"]["off_cpu_before_usage_seconds"]>=e["parent_off_core_sampler_cpu_seconds"]>=0
        def key(p):return p["pid"],p["start_ticks"],p["ppid"]
        before={key(p):p for p in e["before"]};after={key(p):p for p in e["after"]}
        assert len(before)==len(e["before"]) and len(after)==len(e["after"])
        eligible={}
        for k,p in before.items():
            q=after.get(k)
            if q is None or not all(set(v)<=set(row["cpus"]) for x in (p,q) for v in x["thread_affinities"].values()):continue
            assert p["thread_affinities"] and q["thread_affinities"]
            assert 0<=p["cpu_ns"]<=q["cpu_ns"]
            for snapshot in (before,after):
                parents={x["pid"]:x["ppid"] for x in snapshot.values()}
                pid=p["pid"];seen=set()
                while pid!=row["pid"]:
                    assert pid not in seen and pid in parents
                    seen.add(pid);pid=parents[pid]
            eligible[k]=min(p["cpu_ns"],q["cpu_ns"])
        credited={key(p):p["cpu_ns"] for p in e["credited"]}
        assert len(credited)==len(e["credited"]) and credited==eligible
        assert e["cpu_seconds"]==sum(credited.values())/1e9 and math.isfinite(e["cpu_seconds"])
        first,last=row["accounting_start"],row["accounting_end"]
        own=(last["children_cpu_seconds"]-first["children_cpu_seconds"]+e["cpu_seconds"]
             +last["self_cpu_seconds"]-first["self_cpu_seconds"]
             -first["off_cpu_after_usage_seconds"]-last["off_cpu_before_usage_seconds"])
        assert abs(row["command_cpu_seconds"]-own)<1e-8
        assert abs(row["others_cpu"]-max(0,row["selected_busy_seconds"]-own)/max(row["wall_seconds"]*len(row["cpus"]),1e-6))<1e-9
        return []
    except (AssertionError,KeyError,TypeError,ValueError):
        return ["Rust live-worker accounting evidence mismatch"]
