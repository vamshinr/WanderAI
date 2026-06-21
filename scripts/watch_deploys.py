"""Poll Fireworks deployments and emit a line when one becomes READY/FAILED.
Guarded under __main__ so importing it is a no-op (safe for pytest discovery)."""
import os
import time


def main():
    from fireworks import Fireworks
    fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    acct = "vamshinr5899-p0wudhc"
    last = {}
    print("DEPLOY WATCH start", flush=True)
    for _ in range(600):                       # ~50 min at 5s; Monitor bounds lifetime
        try:
            deps = list(fw.deployments.list(account_id=acct))
        except Exception as e:
            print(f"poll error: {type(e).__name__} {e}", flush=True)
            time.sleep(15)
            continue
        cur = {}
        for d in deps:
            did = getattr(d, "name", "?").split("/")[-1]
            model = getattr(d, "base_model", "?").split("/")[-1]
            state = getattr(d, "state", "?")
            cur[did] = (model, state)
            if last.get(did, (None, None))[1] != state:
                print(f"DEPLOY {did} ({model}) -> {state}", flush=True)
                if state == "READY":
                    print(f"READY: {model}  address=accounts/{acct}/models/{model}"
                          f"#accounts/{acct}/deployments/{did}", flush=True)
        for did in last:
            if did not in cur:
                print(f"DEPLOY {did} REMOVED", flush=True)
        last = cur
        time.sleep(15)


if __name__ == "__main__":
    main()
